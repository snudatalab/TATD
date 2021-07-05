'''
Time-Aware Tensor Decomposition for Sparse Tensors

Authors:
    - Dawon Ahn     (dawon@snu.ac.kr)
    - Jun-Gi Jang   (elnino4@snu.ac.kr)
    - U Kang        (ukang@snu.ac.kr)
    - Data Mining Lab at Seoul National University.

File: src/utils.py
    - Contains source code for utility functions.
'''


import os
import torch


def get_path(name, sparse, rrank, window, penalty, scheme, lr, count, exp):
    '''Generate paths for saving training results'''
    
    path=f'out/{scheme}/{name}/training/'
    path1=f'out/{scheme}/{name}/model/'
    path2=f'out/{scheme}/{name}/loss/'

    for p in [path, path1, path2]:
        if not os.path.isdir(p):
            os.makedirs(p)
    
    info = f'result_{sparse}_r_{rrank}_w_{window}_p_{penalty}_lr_{lr}_{count}_{exp}'
    
    train_path = os.path.join(path, info + '.txt')
    model_path = os.path.join(path1, info)
    loss_path = os.path.join(path2, info + '.txt')
        
    best_path = f'out/{scheme}/{name}/best.txt'
    
    if not os.path.exists(best_path):
        with open(best_path, 'w') as f:
            f.write('No.\titers\ttime\tsparse\trank\twindow\tpenalty\tscheme\tlr\trmse\tmae\texp\n')
        f.close()

    return train_path, model_path, loss_path, best_path


def save_checkpoints(model, path):
    '''Save a trained model.'''

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)

