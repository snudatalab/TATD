'''
Time-Aware Tensor Decomposition for Sparse Tensors

Authors:
    - Dawon Ahn     (dawon@snu.ac.kr)
    - Jun-Gi Jang   (elnino4@snu.ac.kr)
    - U Kang        (ukang@snu.ac.kr)
    - Data Mining Lab at Seoul National University.

File: src/als.py
    - Contains source code for least square solution.
'''

import time
import torch
import torch.optim as optim

from tqdm import trange
from tatd import *
from utils import *
from train import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient(model, opt, train, penalty, tmode):
    ''' Train a temporal factor matrix with a given optimizer'''

    opt.zero_grad()
    rec = model(train.indices())
    rloss = (rec - train.values()).pow(2).sum()
    sloss = penalty * model.smooth_reg(tmode)
    loss = rloss + sloss 
    loss.backward()
    opt.step()

    return rloss.detach(), sloss.detach()


def my_khatri_rao(matrices, indices_list, skip_matrix=None):
    ''' Implement a Khatri Rao Product'''

    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]
        
    if skip_matrix == 0:
        indices_list = indices_list[1:] # If skip_matrix is time mode
    elif skip_matrix == 1:
        indices_list = torch.stack([indices_list[0], indices_list[2]])
    else:
        indices_list = indices_list[:-1]

    rank = tl.shape(matrices[0])[1]

    # Compute the Khatri-Rao product for the chosen indices
    sampled_kr = torch.ones((len(indices_list[0]), rank)).to(DEVICE)
    for indices, matrix in zip(indices_list, matrices):
        sampled_kr = sampled_kr*matrix[indices, :]

    return sampled_kr, indices_list


def sparse_least_square(model, train, penalty, mode, rank):
    ''' Implement a least square solution considering only nonzeros'''

    values = train.values()
    indices = train.indices()
    rows = indices[mode].shape[0]
    idx = indices[mode]

    kr_prod, indices_list = my_khatri_rao(model.factors, indices, skip_matrix=mode)

    mat_b = torch.bmm(kr_prod.view(-1, rank, 1), kr_prod.view(-1, 1, rank))
    mat_b2 = torch.zeros((rows, rank, rank), dtype = torch.float).to(DEVICE)
    mat_b2 = mat_b2.scatter_add(0, idx.view(-1, 1, 1).expand(-1, rank, rank), mat_b.float())
    mat_b2 = mat_b2 + torch.stack([torch.eye(rank).to(DEVICE)] * rows) * penalty
    
    vec_c = kr_prod * values.view(-1, 1)
    vec_c2 = torch.zeros((rows, rank), dtype = torch.float).to(DEVICE)
    vec_c2 = vec_c2.scatter_add(0, idx.view(-1, 1).expand(-1, rank), vec_c.float())
    
    factor = torch.bmm(torch.inverse(mat_b2), vec_c2.view(-1, rank, 1)).sum(dim = 2)
    factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
    model.factors[mode].data = factor.to(DEVICE)
    
    

def als_train_model(dataset, model, penalty, opt_scheme, lr, rank,
                    t_path, m_path, l_path, b_path):
    ''' Train a model with ALS+Adam '''

    stop_sign = 3
    name = dataset['name']
    train, valid, test = dataset['train'], dataset['valid'], dataset['test']
    nmode, tmode = dataset['nmode'], dataset['tmode']
    window, count = dataset['window'], dataset['count']

    head = 'Iters\tTime\tTrnRMSE\tTrMAE\tValRMSE\tValMAE\n'
    with open(t_path, 'w') as f:
        f.write(head)

    opt = optim.Adam(model.parameters(), lr = lr,)

    start_time = time.time()
    trn_rmse, val_rmse, rloss, sloss = 0, 0, 0, 0
    old_rmse, inner_rmse, stop_iter, = 1e+5, 1e+5, 0
    c = 0
    with trange(10000) as t:
        for n_iter in t:
            t.set_description(f'trn_rmse : {trn_rmse:.4f} val_rmse : {val_rmse:.4f} rec loss :{rloss:.4f} s loss : {sloss:.4f}')
            inner_num = 0
            stop = True
            for mode in range(nmode):
                if mode != tmode:
                    sparse_least_square(model, train, penalty, mode, rank)
                else:
                    while(stop):
                        rloss, sloss = gradient(model, opt, train, penalty, tmode)
                        trn_rmse, trn_mae = evaluate(model, train)
                        val_rmse, val_mae = evaluate(model, valid)
                        t.set_description(f'trn_rmse : {trn_rmse:.4f} val_rmse : {val_rmse:.4f} rec loss :{rloss:.4f} s loss : {sloss:.4f}')
                        if isNaN(trn_rmse):
                            print("Nan break")
                            break
                        if inner_num > 5:
                            stop = False
                        if val_rmse > inner_rmse:
                            inner_num += 1
                        inner_rmse = val_rmse
            trn_rmse, trn_mae = evaluate(model, train)
            val_rmse, val_mae = evaluate(model, valid)
            t.set_description(f'trn_rmse : {trn_rmse:.4f} val_rmse : {val_rmse:.4f} rec loss :{rloss:.4f} s loss : {sloss:.4f}')

            if val_rmse > old_rmse and n_iter > 10:
                stop_iter += 1
            old_rmse = val_rmse

            with open(t_path, 'a') as f:
                elapsed = time.time() - start_time
                f.write(f'{n_iter:5d}\t{elapsed:.5f}\t')
                f.write(f'{trn_rmse:.5f}\t{trn_mae:.5f}\t')
                f.write(f'{val_rmse:.5f}\t{val_mae:.5f}\n')
            if stop_iter == stop_sign or n_iter == 9999:
                te_rmse, te_mae = evaluate(model, test)
                with open(b_path, 'a') as f1:
                    f1.write(f'{count}\t{n_iter:5d}\t{elapsed:.3f}\t')
                    f1.write(f'{model.sparse}\t')
                    f1.write(f'{model.factors[0].shape[1]:2d}\t')
                    f1.write(f'{window:2d}\t{penalty:.3f}\t')
                    f1.write(f'{opt_scheme}\t{lr:5f}\t')
                    f1.write(f'{te_rmse:.5f}\t{te_mae:.5f}\t')
                    f1.write(f'{model.exp}\n')

                p = f'{m_path}-{n_iter}.pth.tar'
                save_checkpoints(model, p)
                break
