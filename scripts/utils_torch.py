import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from collections import OrderedDict


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
    elif init == 'svd':
        print('init with svd')
        for mode in range(tensor.ndim):
            U, _, _ = torch.linalg.svd(unfold(tensor, mode), full_matrices=False)
            if tensor.shape[mode] < rank: 
                random_part = torch.randn(U.shape[0], rank - tensor.shape[mode], generator=gen, device=device)
                U = torch.cat((U, random_part), axis=1)
            factors.append(U[:, :rank])
    else:
        raise NotImplementedError(init)
        
    return factors


def quantize_tensor_wrapper(qscheme, bits):
    def _wrapper(x, xmin=None, xmax=None):    
        q = 2 ** (bits-1)
        scale_denom = 2*q - 1
        
        if xmin is None: xmin = x.min()
        if xmax is None: xmax = x.max()
        
        if qscheme == torch.per_tensor_symmetric:
            scale = 2 * torch.where(xmin.abs() > xmax, xmin.abs(), xmax) / scale_denom
            zero_point = torch.zeros(scale.shape).int()
                
            return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale
        
        elif qscheme == torch.per_tensor_affine:
            scale = (xmax - xmin) / scale_denom 
            zero_point = (-q - (xmin / scale).int()).int()
            zero_point = torch.clamp(zero_point, -q, q - 1)
            
            return (torch.clamp(torch.round(x / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
    
    return _wrapper


def quantize_tensor(x, qscheme, bits, scale=None):
    q = 2 ** (bits-1)
    scale_denom = 2*q - 1
        
    xmin = x.min()
    xmax = x.max()
        
    if qscheme == torch.per_tensor_symmetric:
        if scale is None:
            scale = 2 * torch.where(xmin.abs() > xmax, xmin.abs(), xmax) / scale_denom

        return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale

    else:
        raise NotImplementedError(qscheme)


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


def admm_iteration(H, U, F, G, max_iter, eps, bits):
    rank = H.shape[1]
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
    for j in range(1, max_iter):
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        H_T = H_.T
        H_prev = H.clone()
        H = quantize_tensor(H_T - U, qscheme=torch.per_tensor_symmetric, bits=bits, scale=None)
        U += H - H_T

        r = torch.sum((H - H_T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U


# no improvement over standart method
def admm_iteration_polish(H, U, F, G, max_iter, eps):
    rank = H.shape[1]
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=G.device), upper=False)
    for j in range(1, max_iter):
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        H_prev = H.clone()
        H = quantize_tensor(H_.T - U)
        
        # polishing
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
        
        U += H - H_.T

        r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U

# takes too long
def admm_iteration_neighs(H, U, F, G, X, neighs, factors, mode, max_iter, eps):
    rank = H.shape[1]
    device = H.device
    rho = torch.trace(G) / rank
    L = torch.linalg.cholesky(G + rho * torch.eye(G.shape[0], device=device), upper=False)
    for j in range(1, max_iter):
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)

        H_prev = H.clone()
        H = quantize_tensor(H_.T - U)
        
        # polishing
        H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)

        # neighbor search
        # bestH = H.clone()
        # bestH_ = H_.clone()
        print('err 1', squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors)))
        factors[mode] = H_.T
        for factor in factors:
            print(factor.shape)
        best_err = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors))
        print('err 2', best_err)

        scale = (H.max() - H.min()) / scale_denom 
        zero_point = (-q - (H.min() / scale).int()).int()
        zero_point = torch.clamp(zero_point, -q, q - 1)

        ks = torch.randint(low=0, high=H.shape[0], size=(neighs,), device=device)
        js = torch.randint(low=0, high=H.shape[1], size=(neighs,), device=device)
        signs = torch.rand(size=(neighs,)) >= 0.5
        print('looking for neighbors')
        for k, j, sign in zip(ks, js, signs):
            print(k,j,sign)
            if sign: H[k,j] += scale
            else: H[k,j] -= scale
            H = quantize_tensor(H, scale=scale, zero_point=zero_point)
            prev_H_ = H_.clone()
            H_ = torch.cholesky_solve((F + rho * (H + U)).T, L, upper=False)
            factors[mode] = H_.T
            # error of convex objective (CP)
            err = squared_relative_diff(X, torch.einsum('ir,jr,kr->ijk', *factors))
            if err < best_err: 
                print('Found a better neighbor with err', err)
                best_err = err
            else:
                if sign: H[k,j] -= scale
                else: H[k,j] += scale
                H_ = prev_H_
        
        U += H - H_.T

        r = torch.sum((H - H_.T)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U


def accuracy(model, dataset_loader, device='cuda', num_classes=1000):
    def one_hot(x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
    
    # Set BN and Droupout to eval regime
    model.eval()

    total_correct = 0

    for (x, y) in tqdm(dataset_loader):
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), num_classes)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():
            out = model(x).cpu().detach().numpy()
            predicted_class = np.argmax(out, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size
    return total_correct / total


def calibrate(model, dataset_loader, device='cuda', num_batches=1000):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Set BN to train regime
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    batch_idx = 0
    for (x, y) in tqdm(dataset_loader):
        if batch_idx >= num_batches:
            break

        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
            
        batch_idx += 1

    return model


def build_cp_layer(rank, factors, bias, cin, cout, kernel_size, padding, stride):
    seq = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(in_channels=cin, out_channels=rank, kernel_size=(1, 1), bias=False)),
      ('conv2', nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=kernel_size,
                          groups=rank, padding=padding, stride=stride, bias=False)),
      ('conv3', nn.Conv2d(in_channels=rank, out_channels=cout, kernel_size=(1, 1), 
                          bias=True if bias is not None else False)),
    ]))

    A,B,C = factors
    f_cout = torch.unsqueeze(torch.unsqueeze(A, 2), 3)
    f_cin = torch.unsqueeze(torch.unsqueeze(B.T, 2), 3)
    f_z = torch.unsqueeze(torch.einsum('hwr->rhw', torch.reshape(C, (*kernel_size, rank))), 1)
    assert seq.conv1.weight.data.shape == f_cin.shape, f'Expected shape: {seq.conv1.weight.data.shape}, but got {f_cin.shape}'
    assert seq.conv2.weight.data.shape == f_z.shape, f'Expected shape: {seq.conv2.weight.data.shape}, but got {f_z.shape}'
    assert seq.conv3.weight.data.shape == f_cout.shape, f'Expected shape: {seq.conv3.weight.data.shape}, but got {f_cout.shape}'
    if bias is not None:
        assert seq.conv3.bias.data.shape == bias.shape, f'Expected shape: {seq.conv3.bias.data.shape}, but got {bias.shape}'
    with torch.no_grad():
        seq.conv1.weight = nn.Parameter(f_cin, requires_grad=True)
        seq.conv2.weight = nn.Parameter(f_z, requires_grad=True)
        seq.conv3.weight = nn.Parameter(f_cout, requires_grad=True)
        if bias is not None:
            seq.conv3.bias = nn.Parameter(bias, requires_grad=True)

    return seq


def build_cp2conv_layer(rank, factors, bias, cin, cout, padding, stride):
    seq = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(in_channels=cin, out_channels=rank, kernel_size=(1, 1), padding=padding, stride=stride, bias=False)),
      ('conv2', nn.Conv2d(in_channels=rank, out_channels=cout, kernel_size=(1, 1), 
                          bias=True if bias is not None else False)),
    ]))

    A,B = factors
    f_cout = torch.unsqueeze(torch.unsqueeze(A, 2), 3)
    f_cin = torch.unsqueeze(torch.unsqueeze(B.T, 2), 3)
    assert seq.conv1.weight.data.shape == f_cin.shape, f'Expected shape: {seq.conv1.weight.data.shape}, but got {f_cin.shape}'
    assert seq.conv2.weight.data.shape == f_cout.shape, f'Expected shape: {seq.conv2.weight.data.shape}, but got {f_cout.shape}'
    if bias is not None:
        assert seq.conv2.bias.data.shape == bias.shape, f'Expected shape: {seq.conv2.bias.data.shape}, but got {bias.shape}'
    with torch.no_grad():
        seq.conv1.weight = nn.Parameter(f_cin, requires_grad=True)
        seq.conv2.weight = nn.Parameter(f_cout, requires_grad=True)
        if bias is not None:
            seq.conv2.bias = nn.Parameter(bias, requires_grad=True)

    return seq


def build_cpfc_layer(rank, factors, bias, fin, fout):
    seq = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(in_features=fin, out_features=rank, bias=False)),
      ('fc2', nn.Linear(in_features=rank, out_features=fout, bias=True if bias is not None else False)),
    ]))

    # fc weight is transposed
    B, A = factors
    assert seq.fc1.weight.data.shape == A.T.shape, f'Expected shape: {seq.fc1.weight.data.shape}, but got {A.T.shape}'
    assert seq.fc2.weight.data.shape == B.shape, f'Expected shape: {seq.fc2.weight.data.shape}, but got {B.shape}'
    if bias is not None:
        assert seq.fc2.bias.data.shape == bias.shape, f'Expected shape: {seq.fc2.bias.data.shape}, but got {bias.shape}'
    with torch.no_grad():
        seq.fc1.weight = nn.Parameter(A.T, requires_grad=True)
        seq.fc2.weight = nn.Parameter(B, requires_grad=True)
        if bias is not None:
            seq.fc2.bias = nn.Parameter(bias, requires_grad=True)

    return seq
