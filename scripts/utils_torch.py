import numpy as np
import torch


def squared_relative_diff(X, Y):
    return torch.sqrt(torch.sum((X - Y)**2) / torch.sum(X**2)).item()

# def squared_relative_diff(X, Y):
#     return (torch.sum((X - Y)**2) / torch.sum(X**2)).item()


def init_factors(tensor, rank, init='svd', device=None, seed=None):
    def unfold(tensor, mode):
        return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    factors = []
    if init == 'random':
        for mode in range(tensor.ndim):
            factors.append(torch.randn(tensor.shape[mode], rank, generator=gen, device=device))
    else:
        raise NotImplementedError(init)
        
    return factors


def quantize_tensor_wrapper(q, scale_denom, dtype):
    def _wrapper(x, scale=None, zero_point=None):
        if scale is None or zero_point is None:
            scale = (x.max() - x.min()) / scale_denom 
            zero_point = (-q - (x.min() / scale).int()).int()
            zero_point = torch.clamp(zero_point, -q, q - 1)
        
        return torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype = dtype).dequantize()
    return _wrapper


q = torch.tensor(2**7).float()
scale_denom = torch.tensor(2**8 - 1).float()
quantize_tensor = quantize_tensor_wrapper(q, scale_denom, dtype=torch.qint8)


def admm_iteration_old(H, U, F, G, max_iter, alpha, eps):
    rank = H.shape[1]
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=H.device))
    for j in range(1, max_iter):
        H_ = torch.linalg.inv(L.T) @ torch.linalg.inv(L) @ (F + rho * (H + U)).T
        
        H_prev = H.clone()
        # argmin of simple r(H)
        new_H = H_.T - U
        new_H_quantized = quantize_tensor(new_H)
        H = (rho * new_H + alpha * new_H_quantized) / (rho + 1.0)
        # H = (rho * new_H + alpha * new_H_quantized) / (rho + alpha)

        U += H - H_.T
        
        # r = torch.linalg.norm(H - H_.T, ord='fro') / torch.linalg.norm(H, ord='fro')
        # s = torch.linalg.norm(H - H_prev, ord='fro') / torch.linalg.norm(U, ord='fro')
        # torch linalg norm outputs sqrt!!
        r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U


def admm_iteration(H, U, F, G, max_iter, eps):
    rank = H.shape[1]
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
    for j in range(1, max_iter):
#         H_ = torch.linalg.inv(L.T) @ torch.linalg.inv(L) @ (F + rho * (H + U)).T
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        H_T = H_.T
        H_prev = H.clone()
        H = quantize_tensor(H_T - U)
        U += H - H_T

        r = torch.sum((H - H_T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U