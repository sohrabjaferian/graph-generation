from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
#     '''

#     :param y_pred:
#     :param y:
#     :param weight_length: how long until the end of sequence shall we add weight
#     :param weight_value: the magnitude that the weight is enhanced
#     :return:
#     '''
#     if has_weight:
#         weight = torch.ones(y.size(0), y.size(1), y.size(2), device=device)
#         # weight = torch.ones(y.size(0),y.size(1),y.size(2))
#         weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
#         weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
#         weight[:,-1*weight_length:,:] = weight_linear
#         loss = F.binary_cross_entropy(y_pred, y, weight=weight.to(device))
#     else:
#         loss = F.binary_cross_entropy(y_pred, y)
#     return loss


def binary_cross_entropy_weight(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    has_weight: bool = False,
    weight_length: int = 1,
    weight_max: float = 10.0,
    reduction: str = "mean",
    # reduction: str = "sum",

) -> torch.Tensor:
    """
    Weighted BCE loss with GPU-efficient weight computation.
    - y_pred, y_true: (B, L, D)
    - if has_weight: linearly upweights last `weight_length` timesteps by up to `weight_max`
    """
    if has_weight:
        B, L, D = y_true.shape
        # Create weights directly on GPU
        w_linear = torch.linspace(1.0, weight_max, weight_length, device=y_true.device)
        # shape [1, weight_length, 1] -> broadcast into last timesteps
        weight = torch.ones((B, L, D), device=y_true.device)
        weight[:, -weight_length:, :] = w_linear.view(1, -1, 1)
        return F.binary_cross_entropy(y_pred, y_true, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(y_pred, y_true, reduction=reduction)



def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size())).to(device)
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size())*thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result

# def gumbel_softmax(logits, temperature, eps=1e-9):
#     '''

#     :param logits: shape: N*L
#     :param temperature:
#     :param eps:
#     :return:
#     '''
#     # get gumbel noise
#     noise = torch.rand(logits.size())
#     noise.add_(eps).log_().neg_()
#     noise.add_(eps).log_().neg_()
#     noise = Variable(noise).to(device)

#     x = (logits + noise) / temperature
#     x = F.softmax(x)
#     return x


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """
    Fast, numerically stable Gumbel-Softmax.
    Draws Gumbel noise directly on GPU.
    Returns differentiable one-hot-like samples.
    """
    # Draw uniform noise directly on device
    u = torch.rand_like(logits)
    # Sample from Gumbel(0,1)
    g = -torch.log(-torch.log(u.clamp(min=eps)))
    # Reparameterize
    x = (logits + g) / temperature
    return F.softmax(x, dim=-1)


# for i in range(10):
#     x = Variable(torch.randn(1,10)).to(device)
#     y = gumbel_softmax(x, temperature=0.01)
#     print(x)
#     print(y)
#     _,id = y.topk(1)
#     print(id)


# def gumbel_sigmoid(logits, temperature):
#     '''

#     :param logits:
#     :param temperature:
#     :param eps:
#     :return:
#     '''
#     # get gumbel noise
#     noise = torch.rand(logits.size()) # uniform(0,1)
#     noise_logistic = torch.log(noise)-torch.log(1-noise) # logistic(0,1)
#     noise = Variable(noise_logistic).to(device)

#     x = (logits + noise) / temperature
#     x = torch.sigmoid(x)
#     return x


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """
    Differentiable binary sample via Gumbel–Sigmoid trick.
    Draws logistic noise directly on GPU for speed and stability.
    """
    # Sample uniform noise directly on the same device
    u = torch.rand_like(logits)
    # Logistic(0,1) noise = log(u) - log(1 - u)
    noise = torch.log(u.clamp(min=eps)) - torch.log((1 - u).clamp(min=eps))
    # Reparameterize and squash
    return torch.sigmoid((logits + noise) / temperature)


# x = Variable(torch.randn(100)).to(device)
# y = gumbel_sigmoid(x,temperature=0.01)
# print(x)
# print(y)

# def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
#     '''
#         do sampling over unnormalized score
#     :param y: input
#     :param sample: Bool
#     :param thresh: if not sample, the threshold
#     :param sampe_time: how many times do we sample, if =1, do single sample
#     :return: sampled result
#     '''

#     # do sigmoid first
#     y = torch.sigmoid(y)
#     # do sampling
#     if sample:
#         if sample_time>1:
#             y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(device)
#             # loop over all batches
#             for i in range(y_result.size(0)):
#                 # do 'multi_sample' times sampling
#                 for j in range(sample_time):
#                     y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(device)
#                     y_result[i] = torch.gt(y[i], y_thresh).float()
#                     if (torch.sum(y_result[i]).data>0).any():
#                         break
#                     # else:
#                     #     print('all zero',j)
#         else:
#             y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(device)
#             y_result = torch.gt(y,y_thresh).float()
#     # do max likelihood based on some threshold
#     else:
#         y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).to(device)
#         y_result = torch.gt(y, y_thresh).float()
#     return y_result

# def sample_sigmoid(y, sample=True, thresh=0.5):
#     """
#     Vectorized version of sample_sigmoid.
#     y: [B, L, D] unnormalized logits
#     """
#     y = torch.sigmoid(y)
#     if sample:
#         # Single vectorized draw from U(0,1)
#         rand = torch.rand_like(y, device=y.device)
#         y_result = (y > rand).float()
#     else:
#         y_result = (y > thresh).float()
#     return y_result


def sample_sigmoid(y, sample=True, temperature=1.0):
    probs = torch.sigmoid(y / temperature)
    if sample:
        # Gumbel-sigmoid reparameterization
        u = torch.rand_like(probs)
        gumbel = -torch.log(-torch.log(u + 1e-12) + 1e-12)
        return torch.sigmoid((torch.log(probs + 1e-12) - torch.log(1 - probs + 1e-12) + gumbel) / temperature)
    else:
        return probs



def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2, device=None):
    """
    Sample adjacency entries under supervision.

    Args:
        y_pred: raw (unnormalized) logits from the network, shape [B, T, F]
        y: ground truth supervision, same shape
        current: current step index
        y_len: tensor/list of lengths for each sequence
        sample_time: number of attempts for multi-sampling when supervision is off
        device: optional torch.device
    Returns:
        y_result: float tensor of sampled (0/1) results
    """

    device = device or (y_pred.device if torch.is_tensor(y_pred) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    y_pred = torch.sigmoid(y_pred)

    # initialize random matrix for result
    y_result = torch.rand_like(y_pred, device=device)

    for i in range(y_result.size(0)):  # batch loop
        if current < y_len[i]:
            # supervised sampling: ensure non-negative diff vs target
            while True:
                y_thresh = torch.rand(y_pred.size(1), y_pred.size(2), device=device)
                sampled = (y_pred[i] > y_thresh).float()
                y_diff = sampled - y[i]
                if torch.all(y_diff >= 0):
                    y_result[i] = sampled
                    break
        else:
            # unsupervised sampling
            for _ in range(sample_time):
                y_thresh = torch.rand(y_pred.size(1), y_pred.size(2), device=device)
                sampled = (y_pred[i] > y_thresh).float()
                if torch.any(torch.sum(sampled) > 0):
                    y_result[i] = sampled
                    break

    return y_result


def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2, device=None):
    """
    Supervised sampling (simplified version).

    Args:
        y_pred: raw (unnormalized) logits [B, T, F]
        y: ground truth tensor [B, T, F]
        current: current time step (int)
        y_len: list or tensor of sequence lengths
        sample_time: number of resampling attempts
        device: optional torch.device

    Returns:
        y_result: sampled 0/1 tensor on the same device
    """

    device = device or (y_pred.device if torch.is_tensor(y_pred) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # normalize logits into probabilities
    y_pred = torch.sigmoid(y_pred)

    # initialize random tensor for outputs
    y_result = torch.rand_like(y_pred, device=device)

    for i in range(y_result.size(0)):
        if current < y_len[i]:
            # directly use ground truth
            y_result[i] = y[i]
        else:
            # unsupervised stochastic sampling
            for _ in range(sample_time):
                y_thresh = torch.rand(y_pred.size(1), y_pred.size(2), device=device)
                sampled = (y_pred[i] > y_thresh).float()
                if torch.sum(sampled) > 0:  # torch.sum returns a scalar tensor
                    y_result[i] = sampled
                    break

    return y_result


################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well)
#####
# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary

# plain LSTM model
# class LSTM_plain(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
#         super(LSTM_plain, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.has_input = has_input
#         self.has_output = has_output

#         if has_input:
#             self.input = nn.Linear(input_size, embedding_size)
#             self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         else:
#             self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         if has_output:
#             self.output = nn.Sequential(
#                 nn.Linear(hidden_size, embedding_size),
#                 nn.ReLU(),
#                 nn.Linear(embedding_size, output_size)
#             )

#         self.relu = nn.ReLU()
#         # initialize
#         self.hidden = None # need initialize before forward run

#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.25)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def init_hidden(self, batch_size):
#         return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device),
#                 Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device))

#     def forward(self, input_raw, pack=False, input_len=None):
#         if self.has_input:
#             input = self.input(input_raw)
#             input = self.relu(input)
#         else:
#             input = input_raw
#         if pack:
#             input = pack_padded_sequence(input, input_len, batch_first=True)
#         output_raw, self.hidden = self.rnn(input, self.hidden)
#         if pack:
#             output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
#         if self.has_output:
#             output_raw = self.output(output_raw)
#         # return hidden state at each time step
#         return output_raw


class LSTM_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 has_input=True, has_output=False, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        emb_in = embedding_size if has_input else input_size
        if has_input:
            self.input = nn.Sequential(
                nn.Linear(input_size, embedding_size),
                nn.ReLU()
            )
        self.rnn = nn.LSTM(
            input_size=emb_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Vectorized, device-aware initialization
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)
                    # set forget-gate bias (2nd quarter) to 1.0 for better training stability
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param[start:end].fill_(1.0)

    def init_hidden(self, batch_size, device=None):
        dev = device or next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev)
        return h, c

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input_raw = self.input(input_raw)
        if pack:
            input_raw = pack_padded_sequence(input_raw, input_len, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(input_raw)
        if pack:
            output = pad_packed_sequence(output, batch_first=True)[0]
        if self.has_output:
            output = self.output(output)
        return output

# # plain GRU model
# class GRU_plain(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
#         super(GRU_plain, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.has_input = has_input
#         self.has_output = has_output

#         if has_input:
#             self.input = nn.Linear(input_size, embedding_size)
#             self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
#                               batch_first=True)
#         else:
#             self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         if has_output:
#             self.output = nn.Sequential(
#                 nn.Linear(hidden_size, embedding_size),
#                 nn.ReLU(),
#                 nn.Linear(embedding_size, output_size)
#             )

#         self.relu = nn.ReLU()
#         # initialize
#         self.hidden = None  # need initialize before forward run

#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.25)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def init_hidden(self, batch_size):
#         return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device)

#     def forward(self, input_raw, pack=False, input_len=None):
#         if self.has_input:
#             input = self.input(input_raw)
#             input = self.relu(input)
#         else:
#             input = input_raw
#         if pack:
#             input = pack_padded_sequence(input, input_len, batch_first=True)
#         output_raw, self.hidden = self.rnn(input, self.hidden)
#         if pack:
#             output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
#         if self.has_output:
#             output_raw = self.output(output_raw)
#         # return hidden state at each time step
#         return output_raw

class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 has_input=True, has_output=False, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        emb_in = embedding_size if has_input else input_size
        if has_input:
            self.input = nn.Sequential(
                nn.Linear(input_size, embedding_size),
                nn.ReLU()
            )

        self.rnn = nn.GRU(
            input_size=emb_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.fill_(0.0)
                    # GRU has reset, update, and new gates; biasing the update gate helps learning stability
                    n = param.size(0)
                    start, end = n // 3, 2 * n // 3
                    param[start:end].fill_(1.0)

    def init_hidden(self, batch_size, device=None):
        dev = device or next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev)

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input_raw = self.input(input_raw)
        if pack:
            input_raw = pack_padded_sequence(input_raw, input_len, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(input_raw)
        if pack:
            output = pad_packed_sequence(output, batch_first=True)[0]
        if self.has_output:
            output = self.output(output)
        return output


# # a deterministic linear output
# class MLP_plain(nn.Module):
#     def __init__(self, h_size, embedding_size, y_size):
#         super(MLP_plain, self).__init__()
#         self.deterministic_output = nn.Sequential(
#             nn.Linear(h_size, embedding_size),
#             nn.ReLU(),
#             nn.Linear(embedding_size, y_size)
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def forward(self, h):
#         y = self.deterministic_output(h)
#         return y

    
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, y_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        return self.net(h)

# # a deterministic linear output, additional output indicates if the sequence should continue grow
# class MLP_token_plain(nn.Module):
#     def __init__(self, h_size, embedding_size, y_size):
#         super(MLP_token_plain, self).__init__()
#         self.deterministic_output = nn.Sequential(
#             nn.Linear(h_size, embedding_size),
#             nn.ReLU(),
#             nn.Linear(embedding_size, y_size)
#         )
#         self.token_output = nn.Sequential(
#             nn.Linear(h_size, embedding_size),
#             nn.ReLU(),
#             nn.Linear(embedding_size, 1)
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def forward(self, h):
#         y = self.deterministic_output(h)
#         t = self.token_output(h)
#         return y,t


class MLP_token_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(inplace=True)
        )
        self.edge_out = nn.Linear(embedding_size, y_size)
        self.token_out = nn.Linear(embedding_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for layer in [self.edge_out, self.token_out]:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(layer.bias)
            for m in self.core:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        z = self.core(h)
        return self.edge_out(z), self.token_out(z)

# # a deterministic linear output (update: add noise)
# class MLP_VAE_plain(nn.Module):
#     def __init__(self, h_size, embedding_size, y_size):
#         super(MLP_VAE_plain, self).__init__()
#         self.encode_11 = nn.Linear(h_size, embedding_size) # mu
#         self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

#         self.decode_1 = nn.Linear(embedding_size, embedding_size)
#         self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
#         self.relu = nn.ReLU()

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def forward(self, h):
#         # encoder
#         z_mu = self.encode_11(h)
#         z_lsgms = self.encode_12(h)
#         # reparameterize
#         z_sgm = z_lsgms.mul(0.5).exp_()
#         eps = Variable(torch.randn(z_sgm.size())).to(device)
#         z = eps*z_sgm + z_mu
#         # decoder
#         y = self.decode_1(z)
#         y = self.relu(y)
#         y = self.decode_2(y)
#         return y, z_mu, z_lsgms


class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super().__init__()
        self.encode_mu = nn.Linear(h_size, embedding_size)
        self.encode_logvar = nn.Linear(h_size, embedding_size)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        mu = self.encode_mu(h)
        logvar = self.encode_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar

# # a deterministic linear output (update: add noise)
# class MLP_VAE_conditional_plain(nn.Module):
#     def __init__(self, h_size, embedding_size, y_size):
#         super(MLP_VAE_conditional_plain, self).__init__()
#         self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
#         self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

#         self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
#         self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
#         self.relu = nn.ReLU()

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def forward(self, h):
#         # encoder
#         z_mu = self.encode_11(h)
#         z_lsgms = self.encode_12(h)
#         # reparameterize
#         z_sgm = z_lsgms.mul(0.5).exp_()
#         eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2))).to(device)
#         z = eps * z_sgm + z_mu
#         # decoder
#         y = self.decode_1(torch.cat((h,z),dim=2))
#         y = self.relu(y)
#         y = self.decode_2(y)
#         return y, z_mu, z_lsgms


class MLP_VAE_conditional_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super().__init__()
        self.encode_mu = nn.Linear(h_size, embedding_size)
        self.encode_logvar = nn.Linear(h_size, embedding_size)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size + h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        mu = self.encode_mu(h)
        logvar = self.encode_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(torch.cat([h, z], dim=-1)), mu, logvar




########### baseline model 1: Learning deep generative model of graphs

class DGM_graphs(nn.Module):
    """
    Deep Generative Model for Graphs (DGMG) core module.

    Implements:
     - Two rounds of message passing (m_uv, f_n)
     - Graph and node embedding functions (f_m, f_init)
     - Gating for aggregation (f_gate, f_gate_init)
     - Decision heads for add-node (f_an) and add-edge (f_ae)
    """

    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size

        # Message passing layers (2 rounds)
        self.m_uv_1 = nn.Linear(h_size * 2, h_size * 2)
        self.f_n_1 = nn.GRUCell(h_size * 2, h_size)

        self.m_uv_2 = nn.Linear(h_size * 2, h_size * 2)
        self.f_n_2 = nn.GRUCell(h_size * 2, h_size)

        # Graph embedding
        self.f_m = nn.Linear(h_size, h_size * 2)
        self.f_gate = nn.Sequential(
            nn.Linear(h_size, 1),
            nn.Sigmoid()
        )

        # New node embedding
        self.f_m_init = nn.Linear(h_size, h_size * 2)
        self.f_gate_init = nn.Sequential(
            nn.Linear(h_size, 1),
            nn.Sigmoid()
        )
        self.f_init = nn.Linear(h_size * 2, h_size)

        # Add-node decision
        self.f_an = nn.Sequential(
            nn.Linear(h_size * 2, 1),
            nn.Sigmoid()
        )

        # Add-edge decision
        self.f_ae = nn.Sequential(
            nn.Linear(h_size * 2, 1),
            nn.Sigmoid()
        )

        # Node scoring
        self.f_s = nn.Linear(h_size * 2, 1)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    m.bias.zero_()



# def message_passing(node_neighbor, node_embedding, model):
#     node_embedding_new = []
#     for i in range(len(node_neighbor)):
#         neighbor_num = len(node_neighbor[i])
#         if neighbor_num > 0:
#             node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
#             node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
#             message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
#             node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
#         else:
#             message_null = Variable(torch.zeros((node_embedding[i].size(0),node_embedding[i].size(1)*2))).to(device)
#             node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
#     node_embedding = node_embedding_new
#     node_embedding_new = []
#     for i in range(len(node_neighbor)):
#         neighbor_num = len(node_neighbor[i])
#         if neighbor_num > 0:
#             node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
#             node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
#             message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
#             node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
#         else:
#             message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2))).to(device)
#             node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
#     return node_embedding_new




def message_passing(node_in, node_out, node_embedding, model):
    """
    Vectorized message passing.
    Args:
        node_in, node_out: lists of neighbor indices for each node (len = N)
        node_embedding: [B, N, H]
        model: has m_uv_1, m_uv_2, and f_n_1 (GRUCell)
    Returns:
        node_embedding_new: [B, N, H]
    """
    device = node_embedding.device
    B, N, H = node_embedding.shape

    # --- adjacency matrices ---
    A_in = torch.zeros(B, N, N, device=device)
    A_out = torch.zeros(B, N, N, device=device)
    for i, (ins, outs) in enumerate(zip(node_in, node_out)):
        if len(ins):
            A_in[:, i, ins] = 1.0
        if len(outs):
            A_out[:, i, outs] = 1.0

    # --- degree-normalized message aggregation ---
    h = node_embedding
    deg_in = A_in.sum(-1, keepdim=True).clamp(min=1.0)
    deg_out = A_out.sum(-1, keepdim=True).clamp(min=1.0)
    msg_in = torch.bmm(A_in, h) / deg_in
    msg_out = torch.bmm(A_out, h) / deg_out

    # --- concatenate and transform ---
    msg_in_cat = torch.cat((msg_in, h), dim=-1)
    msg_out_cat = torch.cat((h, msg_out), dim=-1)
    in_trans = model.m_uv_1(msg_in_cat)
    out_trans = model.m_uv_2(msg_out_cat)
    message = in_trans + out_trans

    # --- update hidden states (GRUCell vectorized) ---
    h_new_flat = model.f_n_1(message.contiguous().view(B * N, -1),
                             h.contiguous().view(B * N, -1))
    return h_new_flat.view(B, N, H)


def calc_graph_embedding(node_embedding_cat, model):
    """
    Compute graph embedding via gated sum over node embeddings.
    node_embedding_cat: [N, H]
    Returns:
        graph_embedding: [1, H*2]
    """
    node_embedding_graph = model.f_m(node_embedding_cat)
    gate = model.f_gate(node_embedding_cat)
    graph_embedding = torch.sum(node_embedding_graph * gate, dim=0, keepdim=True)
    return graph_embedding


def calc_init_embedding(node_embedding_cat, model):
    """
    Compute initial node embedding given current graph state.
    node_embedding_cat: [N, H]
    Returns:
        init_embedding: [1, H]
    """
    node_embedding_init = model.f_m_init(node_embedding_cat)
    gate = model.f_gate_init(node_embedding_cat)
    init_embedding = torch.sum(node_embedding_init * gate, dim=0, keepdim=True)
    return model.f_init(init_embedding)









################################################## code that are NOT used for final version #############


# RNN that updates according to graph structure, new proposed model
class Graph_RNN_structure(nn.Module):
    def __init__(self, hidden_size, batch_size, output_size, num_layers, is_dilation=True, is_bn=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.is_bn = is_bn

        self.relu = nn.ReLU()

        # dilated conv stack for attention over history
        if is_dilation:
            self.conv_block = nn.ModuleList(
                [nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2 ** i, padding=2 ** i)
                 for i in range(num_layers - 1)]
            )
        else:
            self.conv_block = nn.ModuleList(
                [nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1)
                 for i in range(num_layers - 1)]
            )
        self.bn_block = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers - 1)])
        self.conv_out = nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)

        # mean-aggregator transition (GCN-style)
        self.linear_transition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # rolling history of hidden states: each is [B, H, 1]
        self.hidden_all: list[torch.Tensor] = []

        # init
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.fill_(1.0)
                    m.bias.zero_()
                # (GRU path is commented out; add if you re-enable it)

    def _device_dtype(self):
        p = next(self.parameters())
        return p.device, p.dtype

    
    def init_hidden(self, length: Optional[int] = None):
        """Initialize history with ones (kept to match your original)."""
        device, dtype = self._device_dtype()
        if length is None:
            return torch.ones(self.batch_size, self.hidden_size, 1, device=device, dtype=dtype)
        return [torch.ones(self.batch_size, self.hidden_size, 1, device=device, dtype=dtype)
                for _ in range(length)]

    def forward(self, x, teacher_forcing: bool, temperature: float = 0.5,
                bptt: bool = True, bptt_len: int = 20, flexible: bool = True, max_prev_node: int = 100):
        """
        x: [B, 1, output_size]  (binary mask over candidate previous nodes)
        Returns: (x_pred_logits [B,1,output_size], x_pred_sample [B,1,output_size] in {0,1})
        """
        device, dtype = self._device_dtype()

        # ----- 1) build conv features over history -----
        if not self.hidden_all:
            raise RuntimeError("hidden_all is empty. Call self.hidden_all.append(init_state) before forward().")

        # If you really want truncated BPTT on the first element, do it explicitly/safely:
        if bptt and len(self.hidden_all) > bptt_len:
            # detach older history to limit backprop chain
            keep = self.hidden_all[-bptt_len:]
            self.hidden_all = [t.detach() for t in keep]

        hidden_all_cat = torch.cat(self.hidden_all, dim=2)  # [B, H, T_hist]

        h = hidden_all_cat
        for i in range(self.num_layers - 1):
            h = self.conv_block[i](h)
            if self.is_bn:
                h = self.bn_block[i](h)
            h = self.relu(h)
        x_pred_logits = self.conv_out(h)  # [B, 1, T_hist]  (we’ll assume T_hist == output_size)

        # ----- 2) sample (non-differentiable by design) -----
        # Expect sample_tensor to return probabilities or samples in [0,1]. If it returns probs, we threshold below.
        with torch.no_grad():  # this is inference-style sampling; don’t leak grads through it
            probs = torch.sigmoid(x_pred_logits)
            x_pred_sample = sample_tensor(probs, sample=True)  # shape [B,1,T_hist], float in [0,1]
            x_pred_sample_long = (x_pred_sample > 0.5).to(torch.long)  # [B,1,T_hist]

        # ----- 3) choose mask source and aggregate hidden for new node -----
        # Ensure masks are float for multiplication
        if teacher_forcing:
            mask = x.to(dtype)                      # ground truth
            # sum over time dim (prev nodes)
            x_sum = mask.sum(dim=2, keepdim=True)   # [B,1,1]
            hidden_masked = hidden_all_cat * mask   # broadcasting on channel H axis is fine: [B,H,T] * [B,1,T]
        else:
            mask = x_pred_sample.to(dtype)
            x_sum = x_pred_sample_long.sum(dim=2, keepdim=True).to(dtype)  # [B,1,1]
            hidden_masked = hidden_all_cat * mask

        # avoid divide-by-zero; if no active prev nodes, fall back to zeros
        x_sum_safe = x_sum.clamp(min=1.0)
        hidden_new = hidden_masked.sum(dim=2, keepdim=True) / x_sum_safe    # [B,H,1]
        # If x_sum == 0, above gives zeros; that’s fine.

        # transition (linear + ReLU) expects [B,1,H]
        hidden_new = self.linear_transition(hidden_new.permute(0, 2, 1))  # [B,1,H]
        hidden_new = hidden_new.permute(0, 2, 1)                          # [B,H,1]

        # ----- 4) manage history window -----
        if flexible:
            if teacher_forcing:
                # find earliest active index in ground truth mask (over the single time dim)
                # (no .data; use detach() if you must)
                nz = torch.nonzero(x.squeeze(1) > 0, as_tuple=False)  # [K,2] with dims (batch, time)
            else:
                nz = torch.nonzero(x_pred_sample_long.squeeze(1) > 0, as_tuple=False)
            if nz.numel() > 0:
                # min time index across batch
                min_t = nz[:, 1].min().item()
                # trim from the front up to min_t
                self.hidden_all = self.hidden_all[min_t:]
            # then cap by max_prev_node
            if len(self.hidden_all) > max_prev_node:
                self.hidden_all = self.hidden_all[-max_prev_node:]
        else:
            if self.hidden_all:
                self.hidden_all = self.hidden_all[1:]

        # append new hidden
        self.hidden_all.append(hidden_new)

        return x_pred_logits, x_pred_sample

# batch_size = 8
# output_size = 4
# generator = Graph_RNN_structure(hidden_size=16, batch_size=batch_size, output_size=output_size, num_layers=1).to(device)
# for i in range(4):
#     generator.hidden_all.append(generator.init_hidden())
#
# x = Variable(torch.rand(batch_size,1,output_size)).to(device)
# x_pred = generator(x,teacher_forcing=True, sample=True)
# print(x_pred)




# current baseline model, generating a graph by lstm
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Graph_generator_LSTM(nn.Module):
    def __init__(self, feature_size, input_size, hidden_size, output_size, batch_size, num_layers):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # modules
        self.linear_input = nn.Linear(feature_size, input_size)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # proper initialization
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
            elif "weight_hh" in name:
                init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.25)
        for m in [self.linear_input, self.linear_output]:
            init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            init.zeros_(m.bias)

    def init_hidden(self, batch_size=None, device=None):
        """
        Initializes fresh hidden and cell states.
        Returns tuple (h_0, c_0) each of shape [num_layers, batch_size, hidden_size].
        """
        batch_size = batch_size or self.batch_size
        device = device or next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, input_raw, pack=False, lengths=None, hidden=None):
        """
        input_raw: [B, T, feature_size]
        Returns: output [B, T, output_size]
        """
        device = next(self.parameters()).device
        batch_size = input_raw.size(0)

        input_proj = self.relu(self.linear_input(input_raw))

        if pack:
            input_proj = pack_padded_sequence(input_proj, lengths, batch_first=True, enforce_sorted=False)

        # fresh hidden state if none provided
        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size, device=device)
        else:
            # ensure no gradient link to previous sequence
            hidden = (hidden[0].detach(), hidden[1].detach())

        output_raw, hidden = self.lstm(input_proj, hidden)

        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]

        output = self.linear_output(output_raw)
        return output, hidden



import torch
import torch.nn as nn
import torch.nn.init as init

class Graph_generator_LSTM_output_generator(nn.Module):
    def __init__(self, h_size, n_size, y_size, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_size + n_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, y_size),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                init.zeros_(m.bias)

    def forward(self, h, n, temperature=None):
        # Combine hidden and node embeddings
        y_cat = torch.cat((h, n), dim=2)
        y = self.net(y_cat)
        # placeholder: you can re-enable gumbel-sigmoid if using discrete sampling
        # y = gumbel_sigmoid(y, temperature=temperature)
        return y

    
    

class Graph_generator_LSTM_output_discriminator(nn.Module):
    def __init__(self, h_size, y_size, hidden_dim=64, use_sigmoid=True):
        super().__init__()
        layers = [
            nn.Linear(h_size + y_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                init.zeros_(m.bias)

    def forward(self, h, y):
        y_cat = torch.cat((h, y), dim=2)
        return self.net(y_cat)

    
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        # adj: [B, N, N]  or [N, N]
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# vanilla GCN encoder
import torch
import torch.nn as nn
import torch.nn.init as init

class GCN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bn=False, dropout=0.0):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.use_bn = use_bn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Reinitialize both GraphConv layers
        for m in [self.conv1, self.conv2]:
            init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                init.zeros_(m.bias)
        if self.use_bn:
            for bn in [self.bn1, self.bn2]:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)

    def forward(self, x, adj):
        """
        x: [B, N, in_dim]
        adj: [B, N, N] adjacency matrix (normalized or not)
        """
        x = self.conv1(x, adj)
        if self.use_bn:
            # reshape to apply BN over features
            B, N, H = x.shape
            x = self.bn1(x.view(B * N, H)).view(B, N, H)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, adj)
        if self.use_bn:
            B, N, H = x.shape
            x = self.bn2(x.view(B * N, H)).view(B, N, H)
        return x

    
class GCN_decoder(nn.Module):
    def __init__(self, activation="sigmoid", normalize=False):
        """
        activation: 'sigmoid', 'tanh', or None
        normalize: whether to normalize embeddings before reconstruction
        """
        super().__init__()
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        self.normalize = normalize

    def forward(self, x):
        """
        x: [B, N, D] node embeddings
        returns: [B, N, N] reconstructed adjacency
        """
        if self.normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        y = torch.matmul(x, x.transpose(1, 2))
        return self.act(y)



# GCN based graph embedding
# allowing for arbitrary num of nodes
import torch
import torch.nn as nn
import torch.nn.init as init

class GCN_encoder_graph(nn.Module):
    """
    Multi-layer GCN encoder that aggregates per-layer graph-level embeddings
    via max-pooling over nodes.

    Returns: tensor of shape [B, num_layers, output_dim]
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Define layers
        self.conv_first = GraphConv(input_dim, hidden_dim)
        self.conv_blocks = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(max(num_layers - 2, 0))
        ])
        self.conv_last = GraphConv(hidden_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.conv_first, self.conv_last] + list(self.conv_blocks):
            init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            if hasattr(m, "bias") and m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x, adj):
        """
        x: [B, N, input_dim]
        adj: [B, N, N]
        Returns:
            output [B, num_layers, output_dim]
        """
        B = x.size(0)
        out_all = []

        # First layer
        x = self.act(self.conv_first(x, adj))
        x = self.dropout(x)
        out_all.append(torch.max(x, dim=1, keepdim=False)[0].unsqueeze(1))  # [B, 1, H]

        # Intermediate layers
        for conv in self.conv_blocks:
            x = self.act(conv(x, adj))
            x = self.dropout(x)
            out_all.append(torch.max(x, dim=1, keepdim=False)[0].unsqueeze(1))

        # Last layer
        x = self.act(self.conv_last(x, adj))
        out_all.append(torch.max(x, dim=1, keepdim=False)[0].unsqueeze(1))  # [B, 1, out_dim]

        # Stack graph-level features from all layers
        output = torch.cat(out_all, dim=1)  # [B, num_layers, output_dim]
        return output


# x = Variable(torch.rand(1,8,10)).to(device)
# adj = Variable(torch.rand(1,8,8)).to(device)
# model = GCN_encoder_graph(10,10,10).to(device)
# y = model(x,adj)
# print(y.size())


# def preprocess(A):
#     # Get size of the adjacency matrix
#     size = A.size(1)
#     # Get the degrees for each node
#     degrees = torch.sum(A, dim=2)

#     # Create diagonal matrix D from the degrees of the nodes
#     D = Variable(torch.zeros(A.size(0),A.size(1),A.size(2))).to(device)
#     for i in range(D.size(0)):
#         D[i, :, :] = torch.diag(torch.pow(degrees[i,:], -0.5))
#     # Cholesky decomposition of D
#     # D = np.linalg.cholesky(D)
#     # Inverse of the Cholesky decomposition of D
#     # D = np.linalg.inv(D)
#     # Create an identity matrix of size x size
#     # Create A hat
#     # Return A_hat
#     A_normal = torch.matmul(torch.matmul(D,A), D)
#     # print(A_normal)
#     return A_normal



import torch

def preprocess(A, add_self_loops=True, eps=1e-8):
    """
    Symmetric normalization of adjacency matrix:
        A_norm = D^{-1/2} (A + I) D^{-1/2}

    Args:
        A: [B, N, N] batched adjacency tensor (can be weighted)
        add_self_loops: if True, adds identity to A
        eps: small constant to prevent division by zero

    Returns:
        normalized adjacency tensor [B, N, N]
    """
    device = A.device
    if add_self_loops:
        I = torch.eye(A.size(-1), device=device).unsqueeze(0)
        A = A + I

    deg = A.sum(dim=2)  # [B, N]
    deg_inv_sqrt = torch.rsqrt(deg + eps)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # handle isolated nodes

    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, N, N]
    A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
    return A_norm



# a sequential GCN model, GCN with n layers
import torch
import torch.nn as nn
import torch.nn.init as init

class GCN_generator(nn.Module):
    def __init__(self, hidden_dim, normalize=True):
        super().__init__()
        self.conv = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("relu"))
        if hasattr(self.conv, "bias") and self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x, teacher_force=False, adj_real=None):
        """
        Args:
            x: [B, N, F] node features
            teacher_force: if True, use ground-truth adjacency for partial graph
            adj_real: [B, N, N] ground-truth adjacency (needed if teacher_force=True)
        Returns:
            adj_output: [B, N, N] predicted adjacency
        """
        device = x.device
        B, N, _ = x.shape
        adj = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        adj_output = adj.clone()

        # optional normalization of input
        if self.normalize:
            adj = preprocess(adj)

        # initial graph convolution
        x = self.act(self.conv(x, adj))

        for i in range(1, N):
            # ---- 1. compute edge probabilities to previous nodes ----
            x_last = x[:, i:i+1, :]           # [B, 1, F]
            x_prev = x[:, :i, :]              # [B, i, F]
            prob = torch.bmm(x_prev, x_last.transpose(1, 2))  # [B, i, 1]
            # prob = torch.sigmoid(prob)        # ensure in [0,1]
            prob = torch.sigmoid(prob / 2.0)

            # fill symmetric adjacency entries
            adj_output[:, i, :i] = prob.squeeze(-1)
            adj_output[:, :i, i] = prob.squeeze(-1)

            # ---- 2. update adjacency ----
            if teacher_force and adj_real is not None:
                adj[:, :i+1, :i+1] = adj_real[:, :i+1, :i+1]
            else:
                adj[:, i, :i] = prob.squeeze(-1)
                adj[:, :i, i] = prob.squeeze(-1)

            if self.normalize:
                adj = preprocess(adj)

            # ---- 3. update node features ----
            x = self.act(self.conv(x, adj))
            if self.normalize:
                x = x / (x.norm(p=2, dim=2, keepdim=True) + 1e-8)

        return adj_output

# #### test code ####
# print('teacher forcing')
# # print('no teacher forcing')
#
# start = time.time()
# generator = GCN_generator(hidden_dim=4)
# end = time.time()
# print('model build time', end-start)
# for run in range(10):
#     for i in [500]:
#         for batch in [1,10,100]:
#             start = time.time()
#             torch.manual_seed(123)
#             x = Variable(torch.rand(batch,i,4)).to(device)
#             adj = Variable(torch.eye(i).view(1,i,i).repeat(batch,1,1)).to(device)
#             # print('x', x)
#             # print('adj', adj)
#
#             # y = generator(x)
#             y = generator(x,True,adj)
#             # print('y',y)
#             end = time.time()
#             print('node num', i, '  batch size',batch, '  run time', end-start)




import torch
import torch.nn as nn
import torch.nn.init as init

class CNN_decoder(nn.Module):
    def __init__(self, input_size, output_size, stride=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU(inplace=True)

        def make_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=stride),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(out_c, out_c, kernel_size=3, stride=stride),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(out_c, output_size, kernel_size=3, stride=1, padding=1)
            )

        self.block1 = make_block(input_size, input_size // 2, stride)
        self.block2 = make_block(input_size // 2, input_size // 4, stride)
        self.block3 = make_block(input_size // 4, input_size // 8, stride)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, C, L]
        returns: tuple of three decoded feature maps (x_hop1, x_hop2, x_hop3)
        """
        x_hop1 = self.block1(x)
        x_hop2 = self.block2(x_hop1)
        x_hop3 = self.block3(x_hop2)
        return x_hop1, x_hop2, x_hop3

        # # reference code for doing residual connections
        # def _make_layer(self, block, planes, blocks, stride=1):
        #     downsample = None
        #     if stride != 1 or self.inplanes != planes * block.expansion:
        #         downsample = nn.Sequential(
        #             nn.Conv2d(self.inplanes, planes * block.expansion,
        #                       kernel_size=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(planes * block.expansion),
        #         )
        #
        #     layers = []
        #     layers.append(block(self.inplanes, planes, stride, downsample))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes))
        #
        #     return nn.Sequential(*layers)





import torch
import torch.nn as nn
import torch.nn.init as init

class CNN_decoder_share(nn.Module):
    def __init__(self, input_size, output_size, stride=2, hops=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hops = hops

        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=3,
            stride=stride
        )
        self.bn = nn.BatchNorm1d(input_size)
        self.deconv_out = nn.ConvTranspose1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, C, L]
        returns: list of decoded outputs for each hop
        """
        outputs = []
        for _ in range(self.hops):
            # shared feature upsampling
            x = self.relu(self.bn(self.deconv(x)))
            x = self.relu(self.bn(self.deconv(x)))
            # hop output
            outputs.append(self.deconv_out(x))

        return outputs if len(outputs) > 1 else outputs[0]


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNN_decoder_attention(nn.Module):
    def __init__(self, input_size, output_size, stride=2, hops=3, normalize_attention=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hops = hops
        self.normalize_attention = normalize_attention

        self.relu = nn.ReLU(inplace=True)
        self.relu_leaky = nn.LeakyReLU(0.2, inplace=True)

        # shared transposed convolution + batchnorm
        self.deconv = nn.ConvTranspose1d(input_size, input_size, kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(input_size)

        # output and attention heads
        self.deconv_out = nn.ConvTranspose1d(input_size, output_size, kernel_size=3, stride=1, padding=1)
        self.deconv_attention = nn.ConvTranspose1d(input_size, input_size, kernel_size=1, stride=1, padding=0)
        self.bn_attention = nn.BatchNorm1d(input_size)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def compute_attention(self, x_att):
        # x_att: [B, C, L] → attention: [B, L, L]
        x_att = self.relu(self.bn_attention(x_att))
        B, C, L = x_att.shape
        # compute pairwise attention per batch
        att = torch.bmm(x_att.transpose(1, 2), x_att)  # [B, L, L]
        if self.normalize_attention:
            att_norm = att.norm(p=2, dim=(1, 2), keepdim=True) + 1e-8
            att = att / att_norm
        return att

    def forward(self, x):
        """
        x: [B, C, L]
        returns: (decoded_hops, attention_hops)
        """
        x_hops, att_hops = [], []

        for _ in range(self.hops):
            # shared upsampling
            x = self.relu(self.bn(self.deconv(x)))
            x = self.relu(self.bn(self.deconv(x)))

            # decoded feature
            x_out = self.deconv_out(x)
            x_hops.append(x_out)

            # attention
            x_att = self.deconv_attention(x)
            att_map = self.compute_attention(x_att)
            att_hops.append(att_map)

        return x_hops, att_hops






#### test code ####
# x = Variable(torch.randn(1, 256, 1)).to(device)
# decoder = CNN_decoder(256, 16).to(device)
# y = decoder(x)

import torch
import torch.nn as nn
import torch.nn.init as init

class Graphsage_Encoder(nn.Module):
    """
    GraphSAGE-style encoder with hierarchical mean aggregation:
      hop-3 -> hop-2 -> hop-1 -> hop-0 (self),
    then concatenation and projection.

    Inputs:
      nodes_list: list of 4 tensors [hop3, hop2, hop1, hop0], each shaped [B, S_k, F_in]
                  where S_k is the flattened neighbor list length for that hop (per batch).
      nodes_count_list: list of 3 tensors [cnt3, cnt2, cnt1], each shaped [B, P_k]
                  counts of children per parent at that stage (per batch), used to fold S_k -> P_k.

    Output:
      [B, N, 16*input_size]  (same semantics as your original)
    """
    def __init__(self, feature_size, input_size, layer_num=None):
        super().__init__()
        self.input_size = input_size

        # project raw features to model width
        self.linear_projection = nn.Linear(feature_size, input_size)

        # hop 3
        self.linear_3_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_3_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))
        self.linear_3_2 = nn.Linear(input_size * (2 ** 2), input_size * (2 ** 3))

        # hop 2
        self.linear_2_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_2_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))

        # hop 1
        self.linear_1_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))

        # hop 0 (self)
        self.linear_0_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))

        # final concat proj: (2 + 2 + 4 + 8) * input_size  ->  16 * input_size
        self.linear = nn.Linear(input_size * (2 + 2 + 4 + 8), input_size * 16)

        # batch norms for each stage
        self.bn_3_0 = nn.BatchNorm1d(input_size * (2 ** 1))
        self.bn_3_1 = nn.BatchNorm1d(input_size * (2 ** 2))
        self.bn_3_2 = nn.BatchNorm1d(input_size * (2 ** 3))

        self.bn_2_0 = nn.BatchNorm1d(input_size * (2 ** 1))
        self.bn_2_1 = nn.BatchNorm1d(input_size * (2 ** 2))

        self.bn_1_0 = nn.BatchNorm1d(input_size * (2 ** 1))
        self.bn_0_0 = nn.BatchNorm1d(input_size * (2 ** 1))

        # optional final BN (commented in your code); keep shape-compatible if you enable
        # self.bn = nn.BatchNorm1d(input_size * 16)

        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None: init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight); init.zeros_(m.bias)

    @staticmethod
    def _bn_nodewise(x, bn):
        # x: [B, N, F]  -> BN over F
        B, N, F = x.shape
        x = x.reshape(B * N, F)
        x = bn(x)
        return x.reshape(B, N, F)

    @staticmethod
    def _aggregate_by_counts(x, counts):
        """
        Mean-aggregate variable-length child segments into parents per batch.
        x:      [B, S, F]
        counts: [B, P]  (long/int) number of children per parent, row-wise sums = S
        Returns: [B, P, F]
        """
        B, S, F = x.shape
        device = x.device
        out_list = []
        for b in range(B):
            cnt_b = counts[b]  # [P]
            P = cnt_b.numel()
            s = int(cnt_b.sum().item())
            assert s == S, f"Counts sum ({s}) != S ({S}) for batch {b}"
            # split x[b, :S, :] into P chunks
            splits = torch.split(x[b, :S, :], cnt_b.tolist(), dim=0)
            # mean per parent
            means = torch.stack([seg.mean(dim=0) if seg.numel() > 0 else torch.zeros(F, device=device) for seg in splits], dim=0)  # [P, F]
            out_list.append(means)
        return torch.stack(out_list, dim=0)  # [B, P, F]

    def forward(self, nodes_list, nodes_count_list):
        # ---- hop 3 ----
        x3 = self.linear_projection(nodes_list[0])                       # [B, S3, in]
        x3 = self.relu(self._bn_nodewise(self.linear_3_0(x3), self.bn_3_0))  # -> [B, S3, 2*in]
        x3 = self._aggregate_by_counts(x3, nodes_count_list[0])          # [B, P3, 2*in]
        x3 = self.relu(self._bn_nodewise(self.linear_3_1(x3), self.bn_3_1))  # -> [B, P3, 4*in]
        x3 = self._aggregate_by_counts(x3, nodes_count_list[1])          # [B, P2, 4*in]
        x3 = self._bn_nodewise(self.linear_3_2(x3), self.bn_3_2)         # -> [B, P2, 8*in]
        hop3 = x3.mean(dim=1, keepdim=True)                              # [B, 1, 8*in]

        # ---- hop 2 ----
        x2 = self.linear_projection(nodes_list[1])                       # [B, S2, in]
        x2 = self.relu(self._bn_nodewise(self.linear_2_0(x2), self.bn_2_0))  # -> [B, S2, 2*in]
        x2 = self._aggregate_by_counts(x2, nodes_count_list[1])          # [B, P2, 2*in]
        x2 = self._bn_nodewise(self.linear_2_1(x2), self.bn_2_1)         # -> [B, P2, 4*in]
        hop2 = x2.mean(dim=1, keepdim=True)                              # [B, 1, 4*in]

        # ---- hop 1 ----
        x1 = self.linear_projection(nodes_list[2])                       # [B, S1, in]
        x1 = self._bn_nodewise(self.linear_1_0(x1), self.bn_1_0)         # -> [B, S1, 2*in]
        hop1 = x1.mean(dim=1, keepdim=True)                              # [B, 1, 2*in]

        # ---- hop 0 (self) ----
        x0 = self.linear_projection(nodes_list[3])                       # [B, N, in]
        x0 = self._bn_nodewise(self.linear_0_0(x0), self.bn_0_0)         # -> [B, N, 2*in]
        hop0 = x0                                                       # [B, N, 2*in] (no extra mean)

        # ---- concat along feature dim ----
        # hop0: [B, N, 2*in]
        # hop1: [B, 1, 2*in]  (broadcast across N)
        # hop2: [B, 1, 4*in]
        # hop3: [B, 1, 8*in]
        B, N, _ = hop0.shape
        hop1_b = hop1.expand(B, N, hop1.size(-1))
        hop2_b = hop2.expand(B, N, hop2.size(-1))
        hop3_b = hop3.expand(B, N, hop3.size(-1))

        feats = torch.cat([hop0, hop1_b, hop2_b, hop3_b], dim=2)         # [B, N, (2+2+4+8)*in]
        feats = self.linear(feats)                                       # [B, N, 16*in]
        # If you want final BN: uncomment and keep reshape discipline
        # feats = self._bn_nodewise(feats, self.bn)
        return feats
