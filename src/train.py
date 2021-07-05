'''
Time-Aware Tensor Decomposition for Sparse Tensors

Authors:
    - Dawon Ahn     (dawon@snu.ac.kr)
    - Jun-Gi Jang   (elnino4@snu.ac.kr)
    - U Kang        (ukang@snu.ac.kr)
    - Data Mining Lab at Seoul National University.

File: src/train.py
    - Contains source code for training with Adam optimization.
'''


import time
import torch
import torch.functional as F
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from tqdm import trange
from tatd import *
from utils import *

def isNaN(num):
    ''' Find the NaN value'''
    return num != num

def rmse(val, rec):
    ''' Implement RMSE metric'''
    return torch.sqrt(F.mse_loss(val, rec))

def mae(val, rec):
    ''' Implement MAE metric'''
    return torch.mean(torch.abs(val-rec))

def evaluate(model, data):
    ''' Evaluate the model'''
    with torch.no_grad():
        rec = krprod(model.factors, data.indices())
        vals = data.values()
        e1, e2 = rmse(vals, rec), mae(vals, rec)
    return e1.cpu().item(), e2.cpu().item()

        
def training(model, opt, train, penalty, tmode, nmode):
    ''' Train the model with Adam optimizer'''

    for mode in range(nmode):
        model.turn_on_grad(mode)

    opt.zero_grad()

    rec = model(train.indices())

    loss = (rec - train.values()).pow(2).sum()
    
    for n in range(nmode):
        if n == tmode:
            loss = loss + penalty * model.smooth_reg(n)
        else:
            loss = loss + penalty * model.l2_reg(n) 

    loss.backward()

    opt.step()

    return loss

def train_model(dataset, model, penalty, opt_scheme, lr, loss_path, model_path, total_path, exp):
    ''' Train the model with Adam optimizer'''
    
    train, valid, test  = dataset['train'], dataset['valid'], dataset['test']
    nmode, tmode = dataset['nmode'], dataset['tmode']
    window, count = dataset['window'], dataset['count']

    head ='Iters\tTime\tTrnRMSE\tTrMAE\tValRMSE\tValMAE\n'
    
    with open(loss_path, 'w') as f:
        f.write(head)

    if lr == 0.01:
        n_iter_cond = 100
    else:
        n_iter_cond = 200
    if opt_scheme == 'adam':
        opt = optim.Adam(model.parameters(), lr = lr)
    else:
        opt = optim.SGD(model.parameters(), lr = lr)
    
    start_time = time.time()
    old_rmse, stop_iter = 1e+5, 0

    for n_iter in trange(1, 10000):
        loss = training(model, opt, train, penalty, tmode, nmode)
        trn_rmse, trn_mae = evaluate(model, train) 
        val_rmse, val_mae = evaluate(model, valid)
        print(f"Train RMSE : {trn_rmse} Valid RMSE : {val_rmse}")
        
        if isNaN(trn_rmse):
            print("Nan break")
            break
        if val_rmse > old_rmse and n_iter > n_iter_cond:
            stop_iter += 1
        old_rmse = val_rmse
        ttime = time.time() - start_time
        with open(loss_path, 'a') as f:
            elapsed = time.time() - start_time
            f.write(f'{n_iter:5d}\t{elapsed:.5f}\t')
            f.write(f'{trn_rmse:.5f}\t{trn_mae:.5f}\t')
            f.write(f'{val_rmse:.5f}\t{val_mae:.5f}\n')
        if stop_iter == 2 or n_iter > 35000:
            te_rmse, te_mae = evaluate(model, test)
            with open(total_path, 'a') as f1:
                f1.write(f'{exp}\t{count}\t{n_iter:5d}\t{elapsed}\t{model.sparse}\t')
                f1.write(f'{model.factors[0].shape[1]:2d}\t')
                f1.write(f'{window:2d}\t{penalty:.3f}\t')
                f1.write(f'{opt_scheme}\t{lr:5f}\t')
                f1.write(f'{te_rmse:.5f}\t{te_mae:.5f}\n')
                
            p = f'{model_path}-{n_iter}.pth.tar'
            save_checkpoints(model, p)
            break
