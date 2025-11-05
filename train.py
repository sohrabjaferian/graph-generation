import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
import torch
import os
from utils import *
from model import *
from data import *
from args import Args
import create_graphs

from torch.cuda.amp import autocast, GradScaler
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

scaler = GradScaler()

def train_vae_epoch(
    epoch, args, rnn, output, data_loader,
    optimizer_rnn, optimizer_output,
    scheduler_rnn, scheduler_output,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.train()
    output.train()
    loss_sum = 0.0

    for batch_idx, data in enumerate(data_loader):
        # Move batch to GPU once
        x = data["x"].to(device, non_blocking=True, dtype=torch.float32)
        y = data["y"].to(device, non_blocking=True, dtype=torch.float32)
        y_len = data["len"].to(device, non_blocking=True)

        # Sort by sequence length (on GPU)
        y_len_sorted, sort_index = torch.sort(y_len, descending=True)
        x = x[sort_index]
        y = y[sort_index]
        y_len_list = y_len_sorted.tolist()

        # Trim to maximum valid length
        y_len_max = int(y_len_sorted[0])
        x = x[:, :y_len_max, :]
        y = y[:, :y_len_max, :]

        # Initialize hidden state once per batch
        rnn.hidden = rnn.init_hidden(batch_size=x.size(0))

        # -------------------------------
        # Forward + backward with AMP
        # -------------------------------
        optimizer_rnn.zero_grad(set_to_none=True)
        optimizer_output.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            h = rnn(x, pack=True, input_len=y_len_list)
            y_pred, z_mu, z_lsgms = output(h)
            y_pred = torch.sigmoid(y_pred)

            # Repack efficiently
            y_pred, _ = pad_packed_sequence(
                pack_padded_sequence(y_pred, y_len_list, batch_first=True, enforce_sorted=True),
                batch_first=True
            )
            z_mu, _ = pad_packed_sequence(
                pack_padded_sequence(z_mu, y_len_list, batch_first=True, enforce_sorted=True),
                batch_first=True
            )
            z_lsgms, _ = pad_packed_sequence(
                pack_padded_sequence(z_lsgms, y_len_list, batch_first=True, enforce_sorted=True),
                batch_first=True
            )

            # Vectorized BCE + KL divergence
            loss_bce = F.binary_cross_entropy(y_pred, y)
            loss_kl = -0.5 * torch.mean(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
            loss = loss_bce + loss_kl

        # Scaled backward and optimizer steps
        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(output.parameters(), 1.0)
        
        scaler.step(optimizer_rnn)
        scaler.step(optimizer_output)
        scaler.update()

        scheduler_rnn.step()
        scheduler_output.step()

        # -------------------------------
        # Logging
        # -------------------------------
        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"BCE: {loss_bce.item():.6f} | KL: {loss_kl.item():.6f} | "
                f"Graph: {args.graph_type} | Layers: {args.num_layers} | "
                f"Hidden: {args.hidden_size_rnn}"
            )

        loss_sum += loss.item()

    return loss_sum / (batch_idx + 1)



# def train_vae_epoch(epoch, args, rnn, output, data_loader,
#                     optimizer_rnn, optimizer_output,
#                     scheduler_rnn, scheduler_output):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x'].float()
#         y_unsorted = data['y'].float()
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()

#         # if using ground truth to train
#         h = rnn(x, pack=True, input_len=y_len)
#         y_pred,z_mu,z_lsgms = output(h)
#         y_pred = torch.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
#         z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
#         z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
#         z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
#         # use cross entropy loss
#         loss_bce = binary_cross_entropy_weight(y_pred, y)
#         loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
#         loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
#         loss = loss_bce + loss_kl
#         loss.backward()
#         # update deterministic and lstm
#         optimizer_output.step()
#         optimizer_rnn.step()
#         scheduler_output.step()
#         scheduler_rnn.step()


#         z_mu_mean = torch.mean(z_mu.data)
#         z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
#         z_mu_min = torch.min(z_mu.data)
#         z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
#         z_mu_max = torch.max(z_mu.data)
#         z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss_bce.item(), loss_kl.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
#             print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

#         # logging
#         log_value('bce_loss_'+args.fname, loss_bce.item(), epoch*args.batch_ratio+batch_idx)
#         log_value('kl_loss_' +args.fname, loss_kl.item(), epoch*args.batch_ratio + batch_idx)
#         log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
#         log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
#         log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
#         log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
#         log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
#         log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

#         loss_sum += loss.item()
#     return loss_sum/(batch_idx+1)

# def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
#     rnn.hidden = rnn.init_hidden(test_batch_size)
#     rnn.eval()
#     output.eval()

#     # generate graphs
#     max_num_node = int(args.max_num_node)
#     y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#     y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#     x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#     for i in range(max_num_node):
#         h = rnn(x_step)
#         y_pred_step, _, _ = output(h)
#         y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#         # x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
#         x_step = sample_sigmoid(y_pred_step, sample=True)
#         y_pred_long[:, i:i + 1, :] = x_step
#         rnn.hidden = Variable(rnn.hidden.data).cuda()
#     y_pred_data = y_pred.data
#     y_pred_long_data = y_pred_long.data.long()

#     # save graphs as pickle
#     G_pred_list = []
#     for i in range(test_batch_size):
#         adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#         G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#         G_pred_list.append(G_pred)

#     # save prediction histograms, plot histogram over each time step
#     # if save_histogram:
#     #     save_prediction_histogram(y_pred_data.cpu().numpy(),
#     #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
#     #                           max_num_node=max_num_node)


#     return G_pred_list



@torch.no_grad()  # disable grad tracking during inference
def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
    x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)

        # store normalized prediction score
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)

        # sample next step
        x_step = sample_sigmoid(y_pred_step, sample=True)
        y_pred_long[:, i:i + 1, :] = x_step

        # detach hidden state between steps to prevent unwanted gradient buildup
        rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

    y_pred_data = y_pred.cpu()
    y_pred_long_data = y_pred_long.long().cpu()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adjacency matrix
        G_pred_list.append(G_pred)

    # Optional: save prediction histograms
    # if save_histogram:
    #     save_prediction_histogram(
    #         y_pred_data.numpy(),
    #         fname_pred=f"{args.figure_prediction_save_path}{args.fname_pred}{epoch}.jpg",
    #         max_num_node=max_num_node
    #     )

    return G_pred_list


# def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             print('finish node',i)
#             h = rnn(x_step)
#             y_pred_step, _, _ = output(h)
#             y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

#             y_pred_long[:, i:i + 1, :] = x_step
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()

#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


import torch

@torch.no_grad()
def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.eval()
    output.eval()

    G_pred_list = []

    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float().to(device)
        y = data['y'].float().to(device)
        y_len = data['len']
        test_batch_size = x.size(0)

        # initialize hidden state
        rnn.hidden = rnn.init_hidden(test_batch_size)

        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

        for i in range(max_num_node):
            print(f"finish node {i}")
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)

            # normalized prediction scores
            y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)

            # supervised sampling
            x_step = sample_sigmoid_supervised(
                y_pred_step,
                y[:, i:i + 1, :],
                current=i,
                y_len=y_len,
                sample_time=sample_time
            )

            y_pred_long[:, i:i + 1, :] = x_step

            # detach hidden states between iterations
            rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

        y_pred_data = y_pred.cpu()
        y_pred_long_data = y_pred_long.long().cpu()

        # decode predicted adjacency matrices
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].numpy())
            G_pred = get_graph(adj_pred)
            G_pred_list.append(G_pred)

    return G_pred_list


# def train_mlp_epoch(epoch, args, rnn, output, data_loader,
#                     optimizer_rnn, optimizer_output,
#                     scheduler_rnn, scheduler_output):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x'].float()
#         y_unsorted = data['y'].float()
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()

#         h = rnn(x, pack=True, input_len=y_len)
#         y_pred = output(h)
#         y_pred = torch.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         # use cross entropy loss
#         loss = binary_cross_entropy_weight(y_pred, y)
#         loss.backward()
#         # update deterministic and lstm
#         optimizer_output.step()
#         optimizer_rnn.step()
#         scheduler_output.step()
#         scheduler_rnn.step()


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

#         # logging
#         log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

#         loss_sum += loss.item()
#     return loss_sum/(batch_idx+1)



import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.train()
    output.train()
    loss_sum = 0.0

    for batch_idx, data in enumerate(data_loader):
        # zero out gradients safely
        optimizer_rnn.zero_grad(set_to_none=True)
        optimizer_output.zero_grad(set_to_none=True)

        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)

        # trim sequences to maximum valid length
        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        # initialize hidden state
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort by sequence length (descending)
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # forward pass
        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)

        # unpack padded sequences
        y_pred, _ = pad_packed_sequence(
            pack_padded_sequence(y_pred, y_len, batch_first=True),
            batch_first=True
        )

        # compute weighted BCE loss
        loss = binary_cross_entropy_weight(y_pred, y)

        # backward + optimization
        loss.backward()

        # optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(output.parameters(), 5.0)

        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        # log progress
        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(f"Epoch: {epoch}/{args.epochs}, "
                  f"train loss: {loss.item():.6f}, "
                  f"graph type: {args.graph_type}, "
                  f"num_layer: {args.num_layers}, "
                  f"hidden: {args.hidden_size_rnn}")

        log_value(f"loss_{args.fname}", loss.item(), epoch * args.batch_ratio + batch_idx)
        loss_sum += loss.item()

    return loss_sum / (batch_idx + 1)

# def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
#     rnn.hidden = rnn.init_hidden(test_batch_size)
#     rnn.eval()
#     output.eval()

#     # generate graphs
#     max_num_node = int(args.max_num_node)
#     y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#     y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#     x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#     for i in range(max_num_node):
#         h = rnn(x_step)
#         y_pred_step = output(h)
#         y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#         # x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
#         x_step = sample_sigmoid(y_pred_step, sample=True)
#         y_pred_long[:, i:i + 1, :] = x_step
#         rnn.hidden = Variable(rnn.hidden.data).cuda()
#     y_pred_data = y_pred.data
#     y_pred_long_data = y_pred_long.data.long()

#     # save graphs as pickle
#     G_pred_list = []
#     for i in range(test_batch_size):
#         adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#         G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#         G_pred_list.append(G_pred)


#     # # save prediction histograms, plot histogram over each time step
#     # if save_histogram:
#     #     save_prediction_histogram(y_pred_data.cpu().numpy(),
#     #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
#     #                           max_num_node=max_num_node)
#     return G_pred_list


import torch

@torch.no_grad()
def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model states
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
    x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)

        # store normalized prediction score
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)

        # sample binary edges
        x_step = sample_sigmoid(y_pred_step, sample=True)
        y_pred_long[:, i:i + 1, :] = x_step

        # detach hidden states to prevent graph buildup
        rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

    y_pred_data = y_pred.cpu()
    y_pred_long_data = y_pred_long.long().cpu()

    # save generated graphs
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].numpy())
        G_pred = get_graph(adj_pred)  # reconstruct a graph from zero-padded adjacency
        G_pred_list.append(G_pred)

    # optional histogram saving
    # if save_histogram:
    #     save_prediction_histogram(
    #         y_pred_data.numpy(),
    #         fname_pred=f"{args.figure_prediction_save_path}{args.fname_pred}{epoch}.jpg",
    #         max_num_node=max_num_node
    #     )

    return G_pred_list



# def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             print('finish node',i)
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

#             y_pred_long[:, i:i + 1, :] = x_step
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()

#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


import torch

@torch.no_grad()
def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.eval()
    output.eval()

    G_pred_list = []

    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float().to(device)
        y = data['y'].float().to(device)
        y_len = data['len']
        test_batch_size = x.size(0)

        # initialize hidden state
        rnn.hidden = rnn.init_hidden(test_batch_size)

        # prepare containers
        max_num_node = int(args.max_num_node)
        y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

        for i in range(max_num_node):
            print(f"finish node {i}")

            h = rnn(x_step)
            y_pred_step = output(h)

            # normalized sigmoid output
            y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)

            # supervised sampling (with ground truth guidance)
            x_step = sample_sigmoid_supervised(
                y_pred_step,
                y[:, i:i + 1, :],
                current=i,
                y_len=y_len,
                sample_time=sample_time
            )

            y_pred_long[:, i:i + 1, :] = x_step

            # detach hidden state to prevent graph buildup
            rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

        # move predictions to CPU for graph decoding
        y_pred_data = y_pred.cpu()
        y_pred_long_data = y_pred_long.long().cpu()

        # decode predicted adjacency matrices into graphs
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].numpy())
            G_pred = get_graph(adj_pred)
            G_pred_list.append(G_pred)

    return G_pred_list


# def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             print('finish node',i)
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

#             y_pred_long[:, i:i + 1, :] = x_step
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()

#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list



import torch

@torch.no_grad()
def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.eval()
    output.eval()

    G_pred_list = []

    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float().to(device)
        y = data['y'].float().to(device)
        y_len = data['len']
        test_batch_size = x.size(0)

        # initialize RNN hidden state
        rnn.hidden = rnn.init_hidden(test_batch_size)

        # containers
        max_num_node = int(args.max_num_node)
        y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
        x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

        for i in range(max_num_node):
            print(f"finish node {i}")

            # forward pass
            h = rnn(x_step)
            y_pred_step = output(h)

            # normalized output (probabilities)
            y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)

            # supervised sampling (simplified version)
            x_step = sample_sigmoid_supervised_simple(
                y_pred_step,
                y[:, i:i + 1, :],
                current=i,
                y_len=y_len,
                sample_time=sample_time
            )

            y_pred_long[:, i:i + 1, :] = x_step

            # safely detach hidden state to avoid gradient accumulation
            rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

        # move predictions to CPU for graph decoding
        y_pred_data = y_pred.cpu()
        y_pred_long_data = y_pred_long.long().cpu()

        # reconstruct adjacency matrices as graphs
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].numpy())
            G_pred = get_graph(adj_pred)
            G_pred_list.append(G_pred)

    return G_pred_list

# def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x'].float()
#         y_unsorted = data['y'].float()
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()

#         h = rnn(x, pack=True, input_len=y_len)
#         y_pred = output(h)
#         y_pred = torch.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         # use cross entropy loss

#         loss = 0
#         for j in range(y.size(1)):
#             # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
#             end_idx = min(j+1,y.size(2))
#             loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

#         # logging
#         log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

#         loss_sum += loss.item()
#     return loss_sum/(batch_idx+1)


import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.train()
    output.train()

    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()

        # Prepare data
        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len']

        # Truncate to max length in batch
        y_len_max = int(max(y_len_unsorted))
        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        # Initialize hidden state
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # Sort by descending length for packing
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.cpu().numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # Forward through RNN and output network
        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)

        # Unpack sequences
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True, enforce_sorted=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

        # Custom loss: weighted BCE over each sequence step
        loss = 0
        for j in range(y.size(1)):
            end_idx = min(j + 1, y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:, j, :end_idx], y[:, j, :end_idx]) * end_idx

        # Backpropagation
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(output.parameters(), 5.0)
        
        optimizer_output = getattr(args, 'optimizer_output', None)
        optimizer_rnn = getattr(args, 'optimizer_rnn', None)
        if optimizer_output and optimizer_rnn:
            optimizer_output.step()
            optimizer_rnn.step()

        # Logging
        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(
                f"Epoch: {epoch}/{args.epochs}, train loss: {loss.item():.6f}, "
                f"graph type: {args.graph_type}, num_layer: {args.num_layers}, hidden: {args.hidden_size_rnn}"
            )

        log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)
        loss_sum += loss.item()

    return loss_sum / (batch_idx + 1)



## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).cuda()
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


# def train_rnn_epoch(epoch, args, rnn, output, data_loader,
#                     optimizer_rnn, optimizer_output,
#                     scheduler_rnn, scheduler_output):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x'].float()
#         y_unsorted = data['y'].float()
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
#         # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)

#         # input, output for output rnn module
#         # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
#         y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
#         # reverse y_reshape, so that their lengths are sorted, add dimension
#         idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
#         idx = torch.LongTensor(idx)
#         y_reshape = y_reshape.index_select(0, idx)
#         y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

#         output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
#         output_y = y_reshape
#         # batch size for output module: sum(y_len)
#         output_y_len = []
#         output_y_len_bin = np.bincount(np.array(y_len))
#         for i in range(len(output_y_len_bin)-1,0,-1):
#             count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
#             output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
#         # pack into variable
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()
#         output_x = Variable(output_x).cuda()
#         output_y = Variable(output_y).cuda()
#         # print(output_y_len)
#         # print('len',len(output_y_len))
#         # print('y',y.size())
#         # print('output_y',output_y.size())


#         # if using ground truth to train
#         h = rnn(x, pack=True, input_len=y_len)
#         h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
#         # reverse h
#         idx = [i for i in range(h.size(0) - 1, -1, -1)]
#         idx = Variable(torch.LongTensor(idx)).cuda()
#         h = h.index_select(0, idx)
#         hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
#         output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
#         y_pred = output(output_x, pack=True, input_len=output_y_len)
#         y_pred = torch.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
#         output_y = pad_packed_sequence(output_y,batch_first=True)[0]
#         # use cross entropy loss
#         loss = binary_cross_entropy_weight(y_pred, output_y)
#         loss.backward()
#         # update deterministic and lstm
#         optimizer_output.step()
#         optimizer_rnn.step()
#         scheduler_output.step()
#         scheduler_rnn.step()


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

#         # logging
#         log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

        
#         feature_dim = y.size(1)*y.size(2)
#         loss_sum += loss.item()*feature_dim
#     return loss_sum/(batch_idx+1)

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    from torchviz import make_dot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.train()
    output.train()
    loss_sum = 0.0

    for batch_idx, data in enumerate(data_loader):
        # zero gradients
        optimizer_rnn.zero_grad(set_to_none=True)
        optimizer_output.zero_grad(set_to_none=True)

        # prepare inputs
        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len']
        y_len_max = int(max(y_len_unsorted))

        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        # initialize hidden states
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort by sequence length
        y_len, sort_index = torch.sort(y_len_unsorted.to(device), 0, descending=True)
        y_len = y_len.cpu().numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # pack sequences
        y_packed = pack_padded_sequence(y, y_len, batch_first=True, enforce_sorted=True)
        y_reshape = y_packed.data
        # reverse and reshape
        idx = torch.arange(y_reshape.size(0) - 1, -1, -1, device=device)
        y_reshape = y_reshape.index_select(0, idx).view(y_reshape.size(0), y_reshape.size(1), 1)

        # prepare inputs for output network
        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1, device=device),
                              y_reshape[:, :-1, 0:1]), dim=1)
        output_y = y_reshape

        # compute effective lengths for output RNN
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])
            output_y_len.extend([min(i, y.size(2))] * count_temp)

        # forward pass
        h = rnn(x, pack=True, input_len=y_len)
        h_packed = pack_padded_sequence(h, y_len, batch_first=True).data

        # reverse hidden state order
        idx = torch.arange(h_packed.size(0) - 1, -1, -1, device=device)
        h_packed = h_packed.index_select(0, idx)

        hidden_null = torch.zeros(args.num_layers - 1, h_packed.size(0), h_packed.size(1), device=device)
        output.hidden = torch.cat((h_packed.view(1, h_packed.size(0), h_packed.size(1)), hidden_null), dim=0)

        # output pass
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(y_pred)

        # unpack sequences
        y_pred, _ = pad_packed_sequence(pack_padded_sequence(y_pred, output_y_len, batch_first=True),
                                        batch_first=True)
        output_y, _ = pad_packed_sequence(pack_padded_sequence(output_y, output_y_len, batch_first=True),
                                          batch_first=True)
        


        # compute BCE loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(output.parameters(), 5.0)

        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        # logging
        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(f"Epoch: {epoch}/{args.epochs}, train loss: {loss.item():.6f}, "
                  f"graph type: {args.graph_type}, num_layer: {args.num_layers}, hidden: {args.hidden_size_rnn}")

        log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    make_dot(loss, params=dict(list(rnn.named_parameters()) + list(output.named_parameters()))).render("graph_debug", format="png")


    return loss_sum / (batch_idx + 1)


# def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
#     rnn.hidden = rnn.init_hidden(test_batch_size)
#     rnn.eval()
#     output.eval()

#     # generate graphs
#     max_num_node = int(args.max_num_node)
#     y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#     x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#     for i in range(max_num_node):
#         h = rnn(x_step)
#         # output.hidden = h.permute(1,0,2)
#         hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
#         output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
#                                   dim=0)  # num_layers, batch_size, hidden_size
#         x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
#         output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
#         for j in range(min(args.max_prev_node,i+1)):
#             output_y_pred_step = output(output_x_step)
#             # output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
#             output_x_step = sample_sigmoid(output_y_pred_step, sample=True)
#             x_step[:,:,j:j+1] = output_x_step
#             output.hidden = Variable(output.hidden.data).cuda()
#         y_pred_long[:, i:i + 1, :] = x_step
#         rnn.hidden = Variable(rnn.hidden.data).cuda()
#     y_pred_long_data = y_pred_long.data.long()

#     # save graphs as pickle
#     G_pred_list = []
#     for i in range(test_batch_size):
#         adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#         G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#         G_pred_list.append(G_pred)

#     return G_pred_list



# @torch.no_grad()  # inference mode, saves memory and disables grad tracking
# def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     rnn.hidden = rnn.init_hidden(test_batch_size)
#     rnn.eval()
#     output.eval()

#     # generate graphs
#     max_num_node = int(args.max_num_node)
#     y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
#     x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

#     for i in range(max_num_node):
#         h = rnn(x_step)

#         hidden_null = torch.zeros(args.num_layers - 1, h.size(0), h.size(2), device=device)
#         output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)  # num_layers, batch_size, hidden_size

#         x_step = torch.zeros(test_batch_size, 1, args.max_prev_node, device=device)
#         output_x_step = torch.ones(test_batch_size, 1, 1, device=device)

#         for j in range(min(args.max_prev_node, i + 1)):
#             output_y_pred_step = output(output_x_step)
#             output_x_step = sample_sigmoid(output_y_pred_step, sample=True)
#             x_step[:, :, j:j + 1] = output_x_step

#             # Detach hidden states to prevent any unwanted gradient buildup
#             output.hidden = output.hidden.detach()

#         y_pred_long[:, i:i + 1, :] = x_step
#         rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

#     y_pred_long_data = y_pred_long.long().cpu()

#     # save graphs as pickle
#     G_pred_list = []
#     for i in range(test_batch_size):
#         adj_pred = decode_adj(y_pred_long_data[i].numpy())
#         G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
#         G_pred_list.append(G_pred)

#     return G_pred_list



import torch

@torch.no_grad()  # inference mode for efficiency
def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    max_num_node = int(args.max_num_node)

    # containers
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node, device=device)
    x_step = torch.ones(test_batch_size, 1, args.max_prev_node, device=device)

    for i in range(max_num_node):
        # forward through RNN
        h = rnn(x_step)

        # prepare hidden state for output module
        hidden_null = torch.zeros(args.num_layers - 1, h.size(0), h.size(2), device=device)
        output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)

        # autoregressive generation
        x_step = torch.zeros(test_batch_size, 1, args.max_prev_node, device=device)
        output_x_step = torch.ones(test_batch_size, 1, 1, device=device)

        for j in range(min(args.max_prev_node, i + 1)):
            # predict next adjacency column
            output_y_pred_step = output(output_x_step)

            # sample binary edges
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True)

            # assign into growing adjacency matrix
            x_step[:, :, j:j + 1] = output_x_step

            # detach to prevent hidden-state gradient buildup
            output.hidden = output.hidden.detach()

        # record generated adjacency slice
        y_pred_long[:, i:i + 1, :] = x_step

        # detach RNN hidden states between iterations
        rnn.hidden = tuple(h_.detach() for h_ in rnn.hidden)

    # move results to CPU
    y_pred_long_data = y_pred_long.long().cpu()

    # decode adjacency matrices into NetworkX graphs
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].numpy())
        G_pred = get_graph(adj_pred)
        G_pred_list.append(G_pred)

    return G_pred_list

# def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x'].float()
#         y_unsorted = data['y'].float()
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
#         # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)

#         # input, output for output rnn module
#         # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
#         y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
#         # reverse y_reshape, so that their lengths are sorted, add dimension
#         idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
#         idx = torch.LongTensor(idx)
#         y_reshape = y_reshape.index_select(0, idx)
#         y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

#         output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
#         output_y = y_reshape
#         # batch size for output module: sum(y_len)
#         output_y_len = []
#         output_y_len_bin = np.bincount(np.array(y_len))
#         for i in range(len(output_y_len_bin)-1,0,-1):
#             count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
#             output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
#         # pack into variable
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()
#         output_x = Variable(output_x).cuda()
#         output_y = Variable(output_y).cuda()
#         # print(output_y_len)
#         # print('len',len(output_y_len))
#         # print('y',y.size())
#         # print('output_y',output_y.size())


#         # if using ground truth to train
#         h = rnn(x, pack=True, input_len=y_len)
#         h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
#         # reverse h
#         idx = [i for i in range(h.size(0) - 1, -1, -1)]
#         idx = Variable(torch.LongTensor(idx)).cuda()
#         h = h.index_select(0, idx)
#         hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
#         output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
#         y_pred = output(output_x, pack=True, input_len=output_y_len)
#         y_pred = torch.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
#         output_y = pad_packed_sequence(output_y,batch_first=True)[0]
#         # use cross entropy loss
#         loss = binary_cross_entropy_weight(y_pred, output_y)


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

#         # logging
#         log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

#         # print(y_pred.size())
#         feature_dim = y_pred.size(0)*y_pred.size(1)
#         loss_sum += loss.item()*feature_dim/y.size(0)
#     return loss_sum/(batch_idx+1)



import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     rnn.train()
#     output.train()
#     loss_sum = 0.0

#     for batch_idx, data in enumerate(data_loader):
#         # zero out gradients for both modules
#         rnn.zero_grad(set_to_none=True)
#         output.zero_grad(set_to_none=True)

#         x_unsorted = data['x'].float().to(device)
#         y_unsorted = data['y'].float().to(device)
#         y_len_unsorted = data['len']
#         y_len_max = max(y_len_unsorted)

#         # trim to max length
#         x_unsorted = x_unsorted[:, :y_len_max, :]
#         y_unsorted = y_unsorted[:, :y_len_max, :]

#         # initialize hidden state
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

#         # sort sequences by length
#         y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
#         y_len = y_len.tolist()
#         x = torch.index_select(x_unsorted, 0, sort_index)
#         y = torch.index_select(y_unsorted, 0, sort_index)

#         # prepare packed sequences
#         y_packed = pack_padded_sequence(y, y_len, batch_first=True, enforce_sorted=True)
#         y_reshape = y_packed.data

#         # reverse and reshape
#         idx = torch.arange(y_reshape.size(0) - 1, -1, -1, device=device)
#         y_reshape = y_reshape.index_select(0, idx).view(y_reshape.size(0), y_reshape.size(1), 1)

#         # prepare inputs and targets for output RNN
#         output_x = torch.cat(
#             (torch.ones(y_reshape.size(0), 1, 1, device=device), y_reshape[:, :-1, 0:1]),
#             dim=1
#         )
#         output_y = y_reshape

#         # compute length binning for packed output sequences
#         output_y_len = []
#         output_y_len_bin = torch.bincount(torch.tensor(y_len))
#         for i in range(len(output_y_len_bin) - 1, 0, -1):
#             count_temp = int(torch.sum(output_y_len_bin[i:]))
#             output_y_len.extend([min(i, y.size(2))] * count_temp)

#         # forward pass
#         h = rnn(x, pack=True, input_len=y_len)
#         h_packed = pack_padded_sequence(h, y_len, batch_first=True).data

#         # reverse h to align
#         idx = torch.arange(h_packed.size(0) - 1, -1, -1, device=device)
#         h_packed = h_packed.index_select(0, idx)

#         # init output hidden state
#         hidden_null = torch.zeros(args.num_layers - 1, h_packed.size(0), h_packed.size(1), device=device)
#         output.hidden = torch.cat((h_packed.view(1, h_packed.size(0), h_packed.size(1)), hidden_null), dim=0)

#         # predict
#         y_pred = output(output_x, pack=True, input_len=output_y_len)
#         y_pred = torch.sigmoid(y_pred)

#         # unpack for loss computation
#         y_pred, _ = pad_packed_sequence(pack_padded_sequence(y_pred, output_y_len, batch_first=True), batch_first=True)
#         output_y, _ = pad_packed_sequence(pack_padded_sequence(output_y, output_y_len, batch_first=True), batch_first=True)

#         # compute loss
#         loss = binary_cross_entropy_weight(y_pred, output_y)

#         # logging
#         if epoch % args.epochs_log == 0 and batch_idx == 0:
#             print(f"Epoch: {epoch}/{args.epochs}, train loss: {loss.item():.6f}, "
#                   f"graph type: {args.graph_type}, num_layer: {args.num_layers}, hidden: {args.hidden_size_rnn}")

#         log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

#         # backpropagation
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
#         torch.nn.utils.clip_grad_norm_(output.parameters(), 5.0)

#         # update parameters (optimizers assumed to be outside this function)
#         # optimizer_rnn.step()
#         # optimizer_output.step()

#         feature_dim = y_pred.size(0) * y_pred.size(1)
#         loss_sum += loss.item() * feature_dim / y.size(0)

#     return loss_sum / (batch_idx + 1)


import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.train()
    output.train()
    loss_sum = 0.0

    for batch_idx, data in enumerate(data_loader):
        # reset gradients
        rnn.zero_grad(set_to_none=True)
        output.zero_grad(set_to_none=True)

        # data prep
        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len']
        y_len_max = int(max(y_len_unsorted))

        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        # init hidden states
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort by descending length
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # pack and reverse sequences
        y_packed = pack_padded_sequence(y, y_len, batch_first=True, enforce_sorted=True)
        y_reshape = y_packed.data
        idx = torch.arange(y_reshape.size(0) - 1, -1, -1, device=device)
        y_reshape = y_reshape.index_select(0, idx).view(y_reshape.size(0), y_reshape.size(1), 1)

        # prepare output RNN inputs
        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1, device=device), y_reshape[:, :-1, 0:1]),
            dim=1
        )
        output_y = y_reshape

        # construct per-batch length bins
        output_y_len = []
        output_y_len_bin = torch.bincount(torch.tensor(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = int(torch.sum(output_y_len_bin[i:]))
            output_y_len.extend([min(i, y.size(2))] * count_temp)

        # main forward passes
        h = rnn(x, pack=True, input_len=y_len)
        h_packed = pack_padded_sequence(h, y_len, batch_first=True).data
        idx = torch.arange(h_packed.size(0) - 1, -1, -1, device=device)
        h_packed = h_packed.index_select(0, idx)

        hidden_null = torch.zeros(args.num_layers - 1, h_packed.size(0), h_packed.size(1), device=device)
        output.hidden = torch.cat((h_packed.view(1, h_packed.size(0), h_packed.size(1)), hidden_null), dim=0)

        # prediction
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(y_pred)

        # unpack and align for loss
        y_pred, _ = pad_packed_sequence(pack_padded_sequence(y_pred, output_y_len, batch_first=True), batch_first=True)
        output_y, _ = pad_packed_sequence(pack_padded_sequence(output_y, output_y_len, batch_first=True), batch_first=True)

        # compute weighted BCE loss
        loss = binary_cross_entropy_weight(y_pred, output_y)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(output.parameters(), 5.0)

        # no optimizer here; for NLL eval loop they are handled externally
        feature_dim = y_pred.size(0) * y_pred.size(1)
        loss_sum += loss.item() * feature_dim / y.size(0)

        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(f"Epoch: {epoch}/{args.epochs}, train loss: {loss.item():.6f}, "
                  f"graph type: {args.graph_type}, num_layer: {args.num_layers}, hidden: {args.hidden_size_rnn}")

        log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)

    return loss_sum / (batch_idx + 1)



########### train function for LSTM + VAE
# def train(args, dataset_train, rnn, output):
    
#     train_losses = []
    
#     # check if load existing model
#     if args.load:
#         fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
#         rnn.load_state_dict(torch.load(fname))
#         fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
#         output.load_state_dict(torch.load(fname))

#         args.lr = 0.001
#         epoch = args.load_epoch
#         print('model loaded!, lr: {}'.format(args.lr))
#     else:
#         epoch = 1

#     best_loss = float('inf')                 # best loss seen so far
#     patience_counter = 0                     # how many epochs without improvement
#     min_delta = args.min_delta               # minimum change to qualify as improvement
#     patience = args.patience                 # stop after this many bad epochs

#     # initialize optimizer
#     optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
#     optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

#     scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
#     scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

#     # start main loop
#     time_all = np.zeros(args.epochs)
#     while epoch<=args.epochs:
#         time_start = tm.time()
#         # train
#         if 'GraphRNN_VAE' in args.note:
#             loss_sum = train_vae_epoch(epoch, args, rnn, output, dataset_train,
#                                        optimizer_rnn, optimizer_output,
#                                        scheduler_rnn, scheduler_output)
#         elif 'GraphRNN_MLP' in args.note:
#             loss_sum = train_mlp_epoch(epoch, args, rnn, output, dataset_train,
#                                        optimizer_rnn, optimizer_output,
#                                        scheduler_rnn, scheduler_output)
#         elif 'GraphRNN_RNN' in args.note:
#             loss_sum = train_rnn_epoch(epoch, args, rnn, output, dataset_train,
#                                        optimizer_rnn, optimizer_output,
#                                        scheduler_rnn, scheduler_output)

#         train_losses.append(loss_sum)
#         time_end = tm.time()
#         time_all[epoch - 1] = time_end - time_start
#         # test
#         if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
#             for sample_time in range(1,4):
#                 G_pred = []
#                 while len(G_pred)<args.test_total_size:
#                     if 'GraphRNN_VAE' in args.note:
#                         G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
#                     elif 'GraphRNN_MLP' in args.note:
#                         G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
#                     elif 'GraphRNN_RNN' in args.note:
#                         G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
#                     G_pred.extend(G_pred_step)
#                 # save graphs
#                 fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
#                 save_graph_list(G_pred, fname)
#                 if 'GraphRNN_RNN' in args.note:
#                     break
#             print('test done, graphs saved')


#         # save model checkpoint
#         if args.save:
#             if epoch % args.epochs_save == 0:
#                 fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
#                 torch.save(rnn.state_dict(), fname)
#                 fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
#                 torch.save(output.state_dict(), fname)
        
#         # Early Stopping Check
#         # -------------------------------
#         print(f"Epoch {epoch} - Loss: {loss_sum:.6f}, Best: {best_loss:.6f}, Delta: {best_loss - loss_sum:.6f}")
#         if loss_sum < best_loss - min_delta:
#             best_loss = loss_sum
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             print(f" → No improvement. Patience: {patience_counter}/{patience}")
#             if patience_counter >= patience:
#                 print(f"\nEarly stopping at epoch {epoch} (Best loss: {best_loss:.6f})")
#                 break
        
#         epoch += 1
#     np.save(args.timing_save_path + args.fname + "_loss.npy", np.array(train_losses))
#     np.save(args.timing_save_path + args.fname, time_all)

# ########### for graph completion task
# def train_graph_completion(args, dataset_test, rnn, output):
#     fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
#     rnn.load_state_dict(torch.load(fname))
#     fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
#     output.load_state_dict(torch.load(fname))

#     epoch = args.load_epoch
#     print('model loaded!, epoch: {}'.format(args.load_epoch))

#     for sample_time in range(1,4):
#         if 'GraphRNN_MLP' in args.note:
#             G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
#         if 'GraphRNN_VAE' in args.note:
#             G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
#         # save graphs
#         fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
#         save_graph_list(G_pred, fname)
#     print('graph completion done, graphs saved')


import torch
import numpy as np
import time as tm
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

def train(args, dataset_train, rnn, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.to(device)
    output.to(device)

    train_losses = []

    # Load pre-trained checkpoint if requested
    if getattr(args, "load", False):
        rnn_ckpt = f"{args.model_save_path}{args.fname}lstm_{args.load_epoch}.dat"
        output_ckpt = f"{args.model_save_path}{args.fname}output_{args.load_epoch}.dat"
        rnn.load_state_dict(torch.load(rnn_ckpt, map_location=device))
        output.load_state_dict(torch.load(output_ckpt, map_location=device))
        args.lr = 0.001
        epoch = args.load_epoch
        print(f"Model loaded (epoch {epoch}), lr set to {args.lr}")
    else:
        epoch = 1

    # early stopping state
    best_loss = float("inf")
    patience_counter = 0
    min_delta = getattr(args, "min_delta", 1e-4)
    patience = getattr(args, "patience", 10)

    # optimizers + schedulers
    optimizer_rnn = optim.Adam(rnn.parameters(), lr=args.lr)
    optimizer_output = optim.Adam(output.parameters(), lr=args.lr)
    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    time_all = np.zeros(args.epochs)

    while epoch <= args.epochs:
        time_start = tm.time()

        # --- Training Epoch ---
        if "GraphRNN_VAE" in args.note:
            loss_sum = train_vae_epoch(epoch, args, rnn, output, dataset_train,
                                       optimizer_rnn, optimizer_output,
                                       scheduler_rnn, scheduler_output)
        elif "GraphRNN_MLP" in args.note:
            loss_sum = train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                                       optimizer_rnn, optimizer_output,
                                       scheduler_rnn, scheduler_output)
        elif "GraphRNN_RNN" in args.note:
            loss_sum = train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                                       optimizer_rnn, optimizer_output,
                                       scheduler_rnn, scheduler_output)
        else:
            raise ValueError(f"Unknown model type in args.note: {args.note}")

        train_losses.append(loss_sum)
        time_all[epoch - 1] = tm.time() - time_start

        # --- Periodic Testing ---
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    if "GraphRNN_VAE" in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output,
                                                     test_batch_size=args.test_batch_size,
                                                     sample_time=sample_time)
                    elif "GraphRNN_MLP" in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output,
                                                     test_batch_size=args.test_batch_size,
                                                     sample_time=sample_time)
                    elif "GraphRNN_RNN" in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output,
                                                     test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)

                fname = f"{args.graph_save_path}{args.fname_pred}{epoch}_{sample_time}.dat"
                save_graph_list(G_pred, fname)
                if "GraphRNN_RNN" in args.note:
                    break
            print("Test done, graphs saved.")

        # --- Checkpointing ---
        if getattr(args, "save", False) and epoch % args.epochs_save == 0:
            torch.save(rnn.state_dict(), f"{args.model_save_path}{args.fname}lstm_{epoch}.dat")
            torch.save(output.state_dict(), f"{args.model_save_path}{args.fname}output_{epoch}.dat")

        # --- Early Stopping Logic ---
        delta = best_loss - loss_sum
        print(f"Epoch {epoch:03d} | Loss: {loss_sum:.6f} | Best: {best_loss:.6f} | Δ={delta:.6f}")

        if loss_sum < best_loss - min_delta:
            best_loss = loss_sum
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" → No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (Best loss: {best_loss:.6f})")
                break

        epoch += 1

    # save training curves
    np.save(f"{args.timing_save_path}{args.fname}_loss.npy", np.array(train_losses))
    np.save(f"{args.timing_save_path}{args.fname}", time_all)

    

# ########### for NLL evaluation
# def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
#     fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
#     rnn.load_state_dict(torch.load(fname))
#     fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
#     output.load_state_dict(torch.load(fname))

#     epoch = args.load_epoch
#     print('model loaded!, epoch: {}'.format(args.load_epoch))
#     fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
#     with open(fname_output, 'w+') as f:
#         f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
#         f.write('train,test\n')
#         for iter in range(max_iter):
#             if 'GraphRNN_MLP' in args.note:
#                 nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
#                 nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
#             if 'GraphRNN_RNN' in args.note:
#                 nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
#                 nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
#             print('train',nll_train,'test',nll_test)
#             f.write(str(nll_train)+','+str(nll_test)+'\n')

#     print('NLL evaluation done')




@torch.no_grad()
def train_nll(args, dataset_train, dataset_test, rnn, output,
              graph_validate_len, graph_test_len, max_iter=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.to(device)
    output.to(device)

    # --- Load pretrained model ---
    rnn_ckpt = f"{args.model_save_path}{args.fname}lstm_{args.load_epoch}.dat"
    output_ckpt = f"{args.model_save_path}{args.fname}output_{args.load_epoch}.dat"

    rnn.load_state_dict(torch.load(rnn_ckpt, map_location=device))
    output.load_state_dict(torch.load(output_ckpt, map_location=device))

    epoch = args.load_epoch
    print(f"Model loaded (epoch {epoch})")

    # --- Output file setup ---
    os.makedirs(args.nll_save_path, exist_ok=True)
    fname_output = f"{args.nll_save_path}{args.note}_{args.graph_type}.csv"

    with open(fname_output, "w+", encoding="utf-8") as f:
        f.write(f"{graph_validate_len},{graph_test_len}\n")
        f.write("train,test\n")

        for i in range(max_iter):
            if "GraphRNN_MLP" in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            elif "GraphRNN_RNN" in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            else:
                raise ValueError(f"Unsupported model type for NLL evaluation: {args.note}")

            print(f"[Iter {i+1:04d}] Train NLL: {nll_train:.6f} | Test NLL: {nll_test:.6f}")
            f.write(f"{nll_train:.6f},{nll_test:.6f}\n")

    print(f"NLL evaluation done. Results saved to {fname_output}")
