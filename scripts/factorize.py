import sys
if 'tensorly-private' not in sys.path:
    sys.path.append('tensorly-private')

import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
from tensorly.decomposition import parafac
import tensorly as tl

from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import os
import wandb

from utils_torch import *

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--method", dest="method", required=True, type=str,
                    help="admm or parafac")
parser.add_argument("-l", "--layer", dest="layer", required=True, type=str,
                    help="name of a layer to decompose(for example, layer2.1.conv1 means model.layer2[1].conv1)")
parser.add_argument("-r", "--rank", dest="rank", required=True, type=int,
                    help="rank of decomposition")
parser.add_argument("-b", "--bits", dest="bits", required=True, type=int)
parser.add_argument("-s", "--seed", dest="seed", required=True, type=int)

args = parser.parse_args()

bits = args.bits
print('bits:', bits)

# if args.device == 'gpu':
#     device = torch.device('cuda:0')
# elif args.device == 'cpu':
#     device = 'cpu'
# else:
#     raise ValueError('Device can be one of [gpu, cpu]')
device = torch.device('cuda:0')
print('device:', device)
    
orig_model = resnet18(pretrained=True)
orig_model.eval()

if len(args.layer.split('.')) == 1:
    layer = orig_model.__getattr__(args.layer)
elif len(args.layer.split('.')) == 3:
    lname, lidx, ltype = args.layer.split('.')
    lidx = int(lidx)
    layer = orig_model.__getattr__(lname)[lidx].__getattr__(ltype)
    if ltype == 'downsample': layer = layer[0]
else:
    raise ValueError('Invalid layer name')
print('layer:', args.layer)

X = layer.weight.detach().to(device)
bias = layer.bias
if bias is not None: bias = bias.detach().to(device)
if isinstance(layer, nn.Conv2d):
    # kernel_size==1 => equivalent to a Linear layer 
    if len(args.layer.split('.')) == 3 and ltype == 'downsample': 
        X = X.reshape((X.shape[0], X.shape[1]))
    else:
        X = X.reshape((X.shape[0], X.shape[1], -1))

init_method = 'random'
rank = args.rank
print('rank:', rank)

seed = args.seed
print('seed:', seed)

method = args.method
print('method:', method)

outdir = f'{bits}bit_symmetric/factors_{method}_seed{seed}/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

start = time.time()

if method == 'admm':
    run = wandb.init(config=args, 
                     project=f"admm-factorization-{bits}bit-symmetric", 
                     entity="darayavaus",
                     name=f'{method}_{args.layer}_rank{rank}_seed{seed}')
    eps = 1e-8
    tol = 1e-5
    max_iter = 300
    max_iter_factor = 1000
    loss_hist = []
    loss_quant_hist = []
    wandb.config.update({"tol": tol, 
                         "eps": eps,
                         "max_iter": max_iter,
                         "max_iter_factor": max_iter_factor})
    # 3-tensor case
    if X.ndim == 3:
        A, B, C = init_factors(X, rank=rank, init=init_method, device=device, seed=seed)
        U_A = torch.zeros_like(A, device=device)
        U_B = torch.zeros_like(B, device=device)
        U_C = torch.zeros_like(C, device=device)

        for i in tqdm(range(max_iter)):
            G = B.T @ B * (C.T @ C)
            # mttkrp
            F = torch.einsum('abc,cr,br->ar', X, C, B)
            A, U_A = admm_iteration(A, U_A, F, G, max_iter=max_iter_factor, eps=eps, bits=bits)
            A_quantized = quantize_tensor(A, qscheme=torch.per_tensor_symmetric, bits=bits)

            G = A.T @ A * (C.T @ C)
            F = torch.einsum('abc,cr,ar->br', X, C, A)
            B, U_B = admm_iteration(B, U_B, F, G, max_iter=max_iter_factor, eps=eps, bits=bits)
            B_quantized = quantize_tensor(B, qscheme=torch.per_tensor_symmetric, bits=bits)

            G = A.T @ A * (B.T @ B)
            F = torch.einsum('abc,br,ar->cr', X, B, A)
            C, U_C = admm_iteration(C, U_C, F, G, max_iter=max_iter_factor, eps=eps, bits=bits)
            C_quantized = quantize_tensor(C, qscheme=torch.per_tensor_symmetric, bits=bits)

            error = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A, B, C))
            quantized_error = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A_quantized, B_quantized, C_quantized))
            loss_hist.append(error)
            loss_quant_hist.append(quantized_error)
            wandb.log({'Reconstruction Error': error, 
                       'Quantized Reconstruction Error': quantized_error})
            if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < tol: 
                break 
            # safe against exploding error
            if len(loss_hist) > 10 and loss_hist[-1] - loss_hist[-10] > 1e-3:
                break

        factors = [A, B, C]
        factors_quantized = [A_quantized, B_quantized, C_quantized]
    # matrix case
    elif X.ndim == 2:
        A, B = init_factors(X, rank=rank, init=init_method, device=device, seed=seed)
        U_A = torch.zeros_like(A, device=device)
        U_B = torch.zeros_like(B, device=device)
        
        for i in tqdm(range(max_iter)):
            G = B.T @ B
            F = X @ B
            A, U_A = admm_iteration(A, U_A, F, G, max_iter=max_iter_factor, eps=eps, bits=bits)
            A_quantized = quantize_tensor(A, qscheme=torch.per_tensor_symmetric, bits=bits)

            G = A.T @ A
            F = X.T @ A
            B, U_B = admm_iteration(B, U_B, F, G, max_iter=max_iter_factor, eps=eps, bits=bits)
            B_quantized = quantize_tensor(B, qscheme=torch.per_tensor_symmetric, bits=bits)

            error = squared_relative_diff(X, A@B.T)
            quantized_error = squared_relative_diff(X, A_quantized@B_quantized.T)
            loss_hist.append(error)
            loss_quant_hist.append(quantized_error)
            wandb.log({'Reconstruction Error': error, 
                       'Quantized Reconstruction Error': quantized_error})
            if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < tol: 
                break 
            # safe against exploding error
            if len(loss_hist) > 10 and loss_hist[-1] - loss_hist[-10] > 1e-3:
                break
                
            factors = [A, B]
            factors_quantized = [A_quantized, B_quantized]
        
    else: 
        raise ValueError('Incorrect number of dimentions in X')

    torch.save(loss_hist, f'{outdir}/{args.layer}_{method}_{init_method}_rank_{rank}_losshist.pt')
    torch.save(loss_quant_hist, f'{outdir}/{args.layer}_{method}_{init_method}_rank_{rank}_lossquanthist.pt')

elif method == 'parafac':
    tl.set_backend('pytorch')
    _, factors = parafac(X, rank=rank, init=init_method, random_state=seed, 
                         tol=1e-8, stop_criterion='rec_error_deviation', n_iter_max=5000)
    factors_quantized = [quantize_tensor(factor, qscheme=torch.per_tensor_symmetric, bits=bits) for factor in factors]
    
else:
    raise ValueError('Method can be one of [admm, parafac]')

end = time.time()

for mode, factor in enumerate(factors):
    torch.save(factor, f'{outdir}/{args.layer}_{method}_{init_method}_rank_{rank}_mode_{mode}.pt')

if X.ndim == 3:
    print('Factorization took {} minutes'.format((end-start)/60))
    print('Factorization error is {} for usual and {} for quantized'.format(
        squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors)),
        squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors_quantized))
    ))
elif X.ndim == 2:
    pass
else:
    raise ValueError('Incorrect number of dimentions in X')

if method == 'admm': run.finish()
