'''
Time-Aware Tensor Decomposition for Sparse Tensors

Authors:
    - Dawon Ahn     (dawon@snu.ac.kr)
    - Jun-Gi Jang   (elnino4@snu.ac.kr)
    - U Kang        (ukang@snu.ac.kr)
    - Data Mining Lab at Seoul National University.

File: src/read.py
    - Contains source code for reading datasets.
'''


import os
import torch
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_index(length, w1, w2):
    ''' Make dummy indices'''

    return np.arange(-w2, w2+1)[None, :] + np.arange(length+1)[:, None]

def empty_slice(df, tmode):
    ''' Find the empty time slices '''
    
    e_idx = set(np.arange(0, df[tmode].max()+1))
    
    df = df.groupby(tmode).count()
    df = pd.DataFrame(df.iloc[:, 1]).reset_index()
    df.columns = [0,1]
    subset = set(df[0])
    diff = e_idx - subset
    if len(diff) != 0:
        lst = [[i, 0] for i in diff]
        df = df.append(pd.DataFrame(lst))
    df = df.sort_values(by = 0)
    df = df.reset_index(drop = True)
    return df


def cal_time_sparsity(data, tmode):
    ''' Calculate time sparsity '''

    npy = data.indices().cpu().numpy()
    df = pd.DataFrame(npy.T)
    r_index = df[tmode].max()
    r_index = set( np.arange(0, r_index + 1 ))
    df = df.groupby(tmode).count()

    df = pd.DataFrame(df.iloc[:,1])

    df = df.reset_index()
    
    df.columns = [0,1]
    subset = set(df[0])
    diff = r_index - subset
    if len(diff) != 0 :
        lst = [ [i, 1] for i in diff]
        df = df.append(pd.DataFrame(lst))


    df = df.sort_values(by = 0)
    dff = df[1]
    max_, min_ = dff.max(), dff.min()

    min_max = (0.999 - 0.001) * (dff - min_)/(max_ - min_) 
    min_max = np.where(np.isnan(min_max), 1, min_max)
    return 1 - torch.FloatTensor(list(min_max+ 0.001)).to(DEVICE)


def read_tensor(path, name, dtype):
    ''' Read COO format tensor (sparse format) '''

    i =  torch.LongTensor(np.load(os.path.join(path, name, dtype + '_idxs.npy')))
    v = torch.FloatTensor(np.load(os.path.join(path, name, dtype +'_vals.npy')))
    stensor = torch.sparse.FloatTensor(i.t(), v).coalesce()
    return stensor

def read_dataset(name, path = 'data'):
    ''' Read data and make metadata '''
    
    dct = {}
    dct['name'] = name

    for dtype in ['train', 'valid', 'test']:
        dct[dtype] = {}
        stensor = read_tensor(path, name, dtype)
        dct[dtype] = stensor.to(DEVICE)
    
    dct['tmode'] = 0 ### Default value
    dct['nmode'] = len(stensor.shape)
    dct['ndim'] = max(dct['train'].shape, dct['valid'].shape, dct['test'].shape)
    dct['ts'] = cal_time_sparsity(dct['train'], dct['tmode'])

    return dct

    
    
