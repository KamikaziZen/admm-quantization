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

from utils_torch import *

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-l", "--layer", dest="layer", required=True, type=str,
                    help="name of a layer to decompose(for example, layer2.1.conv1 means model.layer2[1].conv1)")
parser.add_argument("-r", "--rank", dest="rank", required=True, type=int,
                    help="rank of decomposition")
parser.add_argument("-m", "--method", dest="method", required=True, type=str,
                    help="admm or parafac")
parser.add_argument("-d", "--device", dest="device", required=True, type=str)

args = parser.parse_args()

if args.device == 'gpu':
    device = torch.device('cuda:0')
elif args.device == 'cpu':
    device = 'cpu'
else:
    raise ValueError('Device can be one of [gpu, cpu]')
print('device:', device)
    
orig_model = resnet18(pretrained=True)
orig_model.eval()

lname, lidx, ltype = args.layer.split('.')
lidx = int(lidx)
layer = orig_model.__getattr__(lname)[lidx].__getattr__(ltype)
print('layer:', layer)

kernel_size = layer.kernel_size
stride = layer.stride
padding = layer.padding
cin = layer.in_channels
cout = layer.out_channels

X = layer.weight.detach().to(device)
X = X.reshape((X.shape[0], X.shape[1], -1))
bias = layer.bias
if bias is not None: bias = bias.detach().to(device)
    
scale = (X.max() - X.min()) / scale_denom
zero_point = (-q - (X.min() / scale).int()).int()
zero_point = torch.clamp(zero_point, -q, q - 1)

init_method = 'random'
rank = args.rank
print('rank:', rank)

start = time.time()

print('method:', args.method)
if args.method == 'admm':
    A, B, C = init_factors(X, rank=rank, init='random', device=device, seed=0)
    U_A = torch.zeros_like(A, device=device)
    U_B = torch.zeros_like(B, device=device)
    U_C = torch.zeros_like(C, device=device)
    eps = 1e-10
    max_iter = 1000
#     alpha = 0.9
    loss_hist = []
    loss_quant_hist = []

    for i in tqdm(range(300)):
        G = B.T @ B * (C.T @ C)
        # mttkrp
        F = torch.einsum('abc,cr,br->ar', X, C, B)
#         A, U_A = admm_iteration(A, U_A, F, G, max_iter=1000, alpha=alpha, eps=eps)
        A, U_A = admm_iteration(A, U_A, F, G, max_iter=1000, eps=eps)

        G = A.T @ A * (C.T @ C)
        F = torch.einsum('abc,cr,ar->br', X, C, A)
#         B, U_B = admm_iteration(B, U_B, F, G, max_iter=1000, alpha=alpha, eps=eps)
        B, U_B = admm_iteration(B, U_B, F, G, max_iter=1000, eps=eps)

        G = A.T @ A * (B.T @ B)
        F = torch.einsum('abc,br,ar->cr', X, B, A)
#         C, U_C = admm_iteration(C, U_C, F, G, max_iter=1000, alpha=alpha, eps=eps)
        C, U_C = admm_iteration(C, U_C, F, G, max_iter=1000, eps=eps)

        A_quantized = quantize_tensor(A, scale=scale, zero_point=zero_point)
        B_quantized = quantize_tensor(B, scale=scale, zero_point=zero_point)
        C_quantized = quantize_tensor(C, scale=scale, zero_point=zero_point)
        loss_quant_hist.append(squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A_quantized, B_quantized, C_quantized)))
        loss_hist.append(squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A, B, C)))
        if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < eps: 
            break
            
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.plot(loss_hist, '.')
    ax1.set_title(f'Loss: {loss_hist[-1]:.5f}, Iterations: {len(loss_hist)}')
    ax2.plot(loss_quant_hist, '.')
    ax2.set_title(f'Loss: {loss_quant_hist[-1]:.5f}, Iterations: {len(loss_quant_hist)}')
    plt.savefig('loss_admm.png')
    
    for mode, factor in enumerate([A,B,C]):
        torch.save(factor, f'factors_proj/{args.layer}_admm_random_rank_{rank}_mode_{mode}.pt')

elif args.method == 'parafac':
    tl.set_backend('pytorch')
    # _, factors = parafac(X, rank=rank, init=init_method, tol=1e-10, n_iter_max=10000, random_state=0)
    # _, factors = parafac(X, rank=rank, init=init_method, stop_criterion='rec_error_deviation', tol=1e-10, n_iter_max=10000,  random_state=0)
    _, factors = parafac(X, rank=rank, init=init_method, random_state=0)
    A,B,C = factors
    A_quantized = quantize_tensor(A)
    B_quantized = quantize_tensor(B)
    C_quantized = quantize_tensor(C)
    
#     for mode, factor in enumerate(factors):
#         torch.save(factor, f'../factors/{layer_name}_parafac_{init_method}_errdev_rank_{rank}_mode_{mode}.pt')
    
    for mode, factor in enumerate(factors):
        torch.save(factor, f'../factors/{args.layer}_parafac_{init_method}_errdev_default_rank_{rank}_mode_{mode}.pt')

else:
    raise NotImplementedError('Invalid method:', args.method)

print(squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A, B, C)),
      squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', A_quantized, B_quantized, C_quantized)))

end = time.time()
print('Factorization took {} minutes'.format((end-start)/60))


