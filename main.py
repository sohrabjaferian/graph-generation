from train import *
import torch
import os
import shutil
import random
from time import gmtime, strftime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if __name__ == '__main__':
    # Load arguments
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)

    # Create directories
    for path in [args.model_save_path, args.graph_save_path, args.figure_save_path,
                 args.timing_save_path, args.figure_prediction_save_path, args.nll_save_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # Logging & TensorBoard setup
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run" + time, flush_secs=5)

    # Load training graphs from a .dat file 
    # train_graphs_path = args.graph_load_path + args.fname_train + '0.dat'

    # === This part to get train data

    train_graphs_path = './real_data/' + args.fname_train + '.dat'
    graphs = load_graph_list(train_graphs_path, is_real=True)
    print(f"Loaded training graphs from: {train_graphs_path}")

    # graphs = load_graph_list(train_graphs_path, is_real=True)
    # print(f"Loaded training graphs from: {train_graphs_path}")

    # Split into train/val/test
    random.seed(123)
    random.shuffle(graphs)

    graphs_len = len(graphs)
    graphs_train = graphs[:int(0.8 * graphs_len)]
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_validate = graphs[:int(0.2 * graphs_len)]

    # Print some statistics
    graph_validate_len = sum(g.number_of_nodes() for g in graphs_validate) / len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = sum(g.number_of_nodes() for g in graphs_test) / len(graphs_test)
    print('graph_test_len', graph_test_len)

    args.max_num_node = max(g.number_of_nodes() for g in graphs)
    max_num_edge = max(g.number_of_edges() for g in graphs)
    min_num_edge = min(g.number_of_edges() for g in graphs)

    print('Total graph num: {}, Training set: {}'.format(len(graphs), len(graphs_train)))
    print('Max number node: {}'.format(args.max_num_node))
    print('Max/Min number edge: {}; {}'.format(max_num_edge, min_num_edge))

    # Save test graphs to .dat file
    # test_graphs_path = args.graph_save_path + args.fname_test + '0.dat'

    test_graphs_path = './test_data/' + args.fname_test + '0.dat'
    save_graph_list(graphs_test, test_graphs_path)
    print('Test graphs saved to:', test_graphs_path)


    # Dataset Initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node - 1
    elif 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train, max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(
            graphs_train,
            max_prev_node=args.max_prev_node,
            max_num_node=args.max_num_node
        )
        if args.max_prev_node is None:
            args.max_prev_node = dataset.max_prev_node
            print(f"[auto] max_prev_node set to {args.max_prev_node}")

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for _ in range(len(dataset))],
        num_samples=args.batch_size * args.batch_ratio,
        replacement=True
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=sample_strategy
    )

    # Model Initialization
    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node,
                        embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn,
                        num_layers=args.num_layers,
                        has_input=True, has_output=False).to(device)
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn,
                                           embedding_size=args.embedding_size_output,
                                           y_size=args.max_prev_node).to(device)
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node,
                        embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn,
                        num_layers=args.num_layers,
                        has_input=True, has_output=False).to(device)
        output = MLP_plain(h_size=args.hidden_size_rnn,
                           embedding_size=args.embedding_size_output,
                           y_size=args.max_prev_node).to(device)
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node,
                        embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn,
                        num_layers=args.num_layers,
                        has_input=True, has_output=True,
                        output_size=args.hidden_size_rnn_output).to(device)
        output = GRU_plain(input_size=1,
                           embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output,
                           num_layers=args.num_layers,
                           has_input=True, has_output=True,
                           output_size=1).to(device)

    # Train
    train(args, dataset_loader, rnn, output)
