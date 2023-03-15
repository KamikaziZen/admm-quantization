import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import tensorly as tl
from tensorly.decomposition import parafac

from source.quantization import quantize_tensor
from source.utils import unfold
from source.parafac_epc import parafac_epc


def squared_relative_diff(X, Y):
    return torch.sqrt(torch.sum((X - Y)**2) / torch.sum(X**2)).item()

# def squared_relative_diff(X, Y):
#     return (torch.sum((X - Y)**2) / torch.sum(X**2)).item()


def init_factors(tensor, rank, init='random', device=None, seed=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    factors = []
    if init == 'random':
        for mode in range(tensor.ndim):
            factors.append(torch.randn(tensor.shape[mode], rank, generator=gen, device=device))
    elif init == 'svd':
        for mode in range(tensor.ndim):
            U, _, _ = torch.linalg.svd(unfold(tensor, mode), full_matrices=False)
            if tensor.shape[mode] < rank: 
                random_part = torch.randn(U.shape[0], rank - tensor.shape[mode], generator=gen, device=device)
                U = torch.cat((U, random_part), axis=1)
            factors.append(U[:, :rank])
    elif init == 'parafac':
        tl.set_backend('pytorch')
        _, factors = parafac(tensor, rank=rank, init='random', random_state=seed,
                             tol=1e-5, n_iter_max=100)
    elif init == 'parafac-epc':
        tl.set_backend('pytorch') 
        _, factors = parafac_epc(tensor, rank=rank, init='random',
                                 als_maxiter=50, epc_maxiter=50)
        factors = [torch.tensor(factor, dtype=torch.float) for factor in factors]
    else:
        raise NotImplementedError(init)
        
    return factors


def admm_iteration(H, U, F, G, max_iter, eps, bits, qscheme):
    rank = H.shape[1]
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
    for j in range(1, max_iter):
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        H_T = H_.T
        H_prev = H.clone()
        H = quantize_tensor(H_T - U, qscheme=qscheme, bits=bits)
        U += H - H_T

        r = torch.sum((H - H_T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U


# def admm_iteration_scaleopt(H, U, F, G, max_iter, eps, bits, qscheme, scale=None):
#     """AO-ADMM iteration step with quantization scale optimization"""
#     rank = H.shape[1]
#     rho = torch.trace(G) / rank
#     L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
#     q = 2 ** bits - 1
#     for j in range(1, max_iter):
        
#         H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
#         H_T = H_.T
        
#         H_prev = H.clone()
#         if scale is None:
#             min_val, max_val = H_.min(), H_.max()
#             scale = 2 * max(min_val.abs(), max_val.abs()) / q
#             print('setting scale:', min_val, max_val, q, scale)
#         print('start iteration')
#         for j in range(10):
#             H_int = torch.floor((H_T - U) / scale + 0.5)
#             H_int = torch.clamp(H_int, -q, q-1)
#             scale = (H_T - U).T @ H_int / (H_int.T @ H_int)
#             # getting rid of + inf and - inf (result of dividing by zero)
#             # scale.double() because of older torch version
#             # newer versions should be fine with 
#             # torch.where(scale.abs() == torch.inf, torch.nan, scale)
#             scale = torch.where(scale.abs() == torch.inf, torch.nan, scale.double()).nanmean().float()
#             print('scale:', scale)
#         H = H_int * scale
            
#         U += H - H_T

#         r = torch.sum((H - H_T)**2) / torch.sum(H**2)
#         s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
#         if r < eps and s < eps: 
#             break
            
#     return H, U, scale
        

# def admm_iteration_old(H, U, F, G, max_iter, alpha, eps):
#     rank = H.shape[1]
#     rho = torch.trace(G) / rank
#     L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=H.device))
#     for j in range(1, max_iter):
#         H_ = torch.linalg.inv(L.T) @ torch.linalg.inv(L) @ (F + rho * (H + U)).T
        
#         H_prev = H.clone()
#         # argmin of simple r(H)
#         new_H = H_.T - U
#         new_H_quantized = quantize_tensor(new_H)
#         H = (rho * new_H + alpha * new_H_quantized) / (rho + 1.0)
#         # H = (rho * new_H + alpha * new_H_quantized) / (rho + alpha)

#         U += H - H_.T
        
#         # r = torch.linalg.norm(H - H_.T, ord='fro') / torch.linalg.norm(H, ord='fro')
#         # s = torch.linalg.norm(H - H_prev, ord='fro') / torch.linalg.norm(U, ord='fro')
#         # torch linalg norm outputs sqrt!!
#         r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
#         s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
#         if r < eps and s < eps: 
#             break
            
#     return H, U


# no improvement over standart method
# def admm_iteration_polish(H, U, F, G, max_iter, eps):
#     rank = H.shape[1]
#     rho = torch.trace(G) / rank
#     L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
#     for j in range(1, max_iter):
#         H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
#         H_prev = H.clone()
#         H = quantize_tensor(H_.T - U)
        
#         # polishing
#         H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        
#         U += H - H_.T

#         r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
#         s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
#         if r < eps and s < eps: 
#             break
            
#     return H, U

# # takes too long
# def admm_iteration_neighs(H, U, F, G, X, neighs, factors, mode, max_iter, eps):
#     rank = H.shape[1]
#     device = H.device
#     rho = torch.trace(G) / rank
#     L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=device), upper=False)
#     for j in range(1, max_iter):
#         H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)

#         H_prev = H.clone()
#         H = quantize_tensor(H_.T - U)
        
#         # polishing
#         H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)

#         # neighbor search
#         # bestH = H.clone()
#         # bestH_ = H_.clone()
#         print('err 1', squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors)))
#         factors[mode] = H_.T
#         for factor in factors:
#             print(factor.shape)
#         best_err = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors))
#         print('err 2', best_err)

#         scale = (H.max() - H.min()) / scale_denom 
#         zero_point = (-q - (H.min() / scale).int()).int()
#         zero_point = torch.clamp(zero_point, -q, q - 1)

#         ks = torch.randint(low=0, high=H.shape[0], size=(neighs,), device=device)
#         js = torch.randint(low=0, high=H.shape[1], size=(neighs,), device=device)
#         signs = torch.rand(size=(neighs,)) >= 0.5
#         print('looking for neighbors')
#         for k, j, sign in zip(ks, js, signs):
#             print(k,j,sign)
#             if sign: H[k,j] += scale
#             else: H[k,j] -= scale
#             H = quantize_tensor(H, scale=scale, zero_point=zero_point)
#             prev_H_ = H_.clone()
#             H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
#             factors[mode] = H_.T
#             # error of convex objective (CP)
#             err = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors))
#             if err < best_err: 
#                 print('Found a better neighbor with err', err)
#                 best_err = err
#             else:
#                 if sign: H[k,j] -= scale
#                 else: H[k,j] += scale
#                 H_ = prev_H_
        
#         U += H - H_.T

#         r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
#         s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
#         if r < eps and s < eps: 
#             break
            
#     return H, U

