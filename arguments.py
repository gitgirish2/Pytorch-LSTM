import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='Walmart Replenishment ISD rnn')
    #parser.add_argument('--save-dir', type=str, help='main folder for saving models')
    parser.add_argument('--data-dir', type=str,default='/home/app/replen_data/', help='folder for storing data')
    parser.add_argument('--save-dir', type=str,default='/home/app/replen_data/',help='folder for saving data')
    #rnn
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--size-s', type=int, default=128,
                        help='rnn hidden state size (default: 128)')
    parser.add_argument('--rnn-layers', type=int, default=1,
                        help='rnn layers (default: 1)')
    parser.add_argument('--seq-len', type=int, default=14,
                        help='warm_up/prediction window lenght, note that the effective size of the seq_len is going to be 2x')
    parser.add_argument('--in-rnn', type=int, default=32,
                        help='to go from MLP to concatenate with the embeddings for the RNN')
    parser.add_argument('--out-rnn', type=int, default=32,
                        help='output of RNN')
    parser.add_argument('--out-mlp', type=int, default=32,
                        help='output of the MLP model')
    parser.add_argument('--embs-size', type=int, default=8,
                        help='the embedding size')

    print(args_string)
    args = parser.parse_args(args=args_string)
    return args
