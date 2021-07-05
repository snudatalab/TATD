'''
Time-Aware Tensor Decomposition for Sparse Tensors

Authors:
    - Dawon Ahn     (dawon@snu.ac.kr)
    - Jun-Gi Jang   (elnino4@snu.ac.kr)
    - U Kang        (ukang@snu.ac.kr)
    - Data Mining Lab at Seoul National University.

File: src/main.py
    - Contains source code for running TATD.
'''


import argparse
from als import *
from read import *
from train import *
from utils import *

torch.manual_seed(1234)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tatd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='indoor', help=u"Dataset")
    parser.add_argument('--sparse', type=float, default=1, help=u"Sparsity penalty")
    parser.add_argument('--rank', type=int, default=10, help=u"Size of factor matrix")
    parser.add_argument('--window', type=int, default=3, help=u"Window size")
    parser.add_argument('--penalty', type=float, default=100, help=u"Strength of penalty")
    parser.add_argument('--scheme', type=str, default='als_adam', help=u"Optimizer scheme")
    parser.add_argument('--lr', type=float, default='0.01', help=u"Learning rate")
    parser.add_argument('--count', type=str, default=1, help=u"Experiment for average")
    parser.add_argument('--exp', type=str, default="default", help=u"Experiment type")
    args = parser.parse_args()

    name = args.name
    rank = args.rank
    window = args.window
    penalty = args.penalty
    sparse = args.sparse
    lr = args.lr
    scheme = args.scheme
    count = args.count
    exp = args.exp

    return name, sparse, rank, window, penalty, scheme, lr, count, exp

def main():

    name, sparse, rank, window, penalty, scheme, lr, count, exp = tatd_parser()

    t_path, m_path, l_path, b_path = get_path(name, sparse, rank, window, penalty,
                                                scheme, lr, count, exp)

    dataset = read_dataset(name)
    dataset['count'], dataset['window'] = count, window
    nmode, ndim, tmode =  dataset['nmode'], dataset['ndim'], dataset['tmode']
    density = dataset['ts']

    model = Tatd(nmode, ndim, tmode, density, rank, window, sparse, exp).to(DEVICE)


    print('---------------------------------------------------------------------------')
    print(f'Dataset : {name:6}')
    print(f'Scheme : {scheme:8}\tLr: {lr}\tSparse: {sparse}')
    print(f'Rank    : {rank:6}\tWindow: {window:8}\tPenalty: {penalty}\t{exp}')
    print('---------------------------------------------------------------------------')

    als_train_model(dataset, model, penalty, scheme, lr, rank, 
                    t_path, m_path, l_path, b_path)


if __name__ == '__main__':
    main()
        
        
