import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
from tensorly.decomposition import parafac
import tensorly as tl

from tqdm import tqdm
from collections import OrderedDict
import time
import os
import random

from source.quantization import quantize_tensor
from source.admm import admm_iteration, init_factors, squared_relative_diff
from source.parafac_epc import parafac_epc

from argparse import ArgumentParser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = [args.method]
    run_name.append(f"l={args.layer}")
    run_name.append(f"r={args.rank}")
    run_name.append(f"b={args.bits}")
    run_name.append(f"s={args.seed}")
    run_name.append(f"i={args.init}")
    run_name.append(f"{args.qscheme}")
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--with_wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--method", 
                        type=str, 
                        required=True,
                        help="[admm, parafac]")
    parser.add_argument("--init",
                        type=str,
                        required=False,
                        default='random',
                        help="[random, parafac, parafac-epc]")
    parser.add_argument("--layer", 
                        type=str, 
                        required=True,
                        help="Name of a layer to decompose(for example, layer2.1.conv1 means model.layer2[1].conv1)")
    parser.add_argument("--rank", 
                        type=int, 
                        required=True,
                        help="Rank for decomposition.")
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        help="Number of quantization bits.")
    parser.add_argument("--max_iter_als",
                        required=True, 
                        type=int,
                        help="Maximum number of iterations of ALS.")
    parser.add_argument("--seed",
                        required=True, 
                        type=int,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_symmetric, tensor_affine, tensor_mse, tensor_minmaxlog]")

    args = parser.parse_args()
    # Args Check
#     if args.method not in ['admm', 'parafac']:
#         raise ValueError('Method must be on of [admm, parafac].')
#     if args.qscheme not in ['tensor_symmetric', 'tensor_affine', 'tensor_mse', 'tensor_minmaxlog']:
#         raise ValueError('Qscheme must be one of [tensor_symmetric, tensor_affine, tensor_mse, tensor_minmaxlog].')
    return args


def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print('Running on:', device)
    
    args = parse_args()
    print('Args:', args)
    
    set_seed(args.seed)
    
    if args.with_wandb:
        import wandb
        run = wandb.init(config=args, 
                         name=run_name(args))
    else:
        run=None
        
    model = resnet18(pretrained=True)
    model.eval()

    if len(args.layer.split('.')) == 1:
        layer = model.__getattr__(args.layer)
    elif len(args.layer.split('.')) == 3:
        lname, lidx, ltype = args.layer.split('.')
        lidx = int(lidx)
        layer = model.__getattr__(lname)[lidx].__getattr__(ltype)
        if ltype == 'downsample': layer = layer[0]
    else:
        raise ValueError('Invalid layer name.')

    weight = layer.weight.detach().to(device)
    bias = layer.bias
    if bias is not None: bias = bias.detach().to(device)
    if isinstance(layer, nn.Conv2d):
        # kernel_size==1 => equivalent to a Linear layer 
        if len(args.layer.split('.')) == 3 and ltype == 'downsample': 
            weight = weight.reshape((weight.shape[0], weight.shape[1]))
        else:
            weight = weight.reshape((weight.shape[0], weight.shape[1], -1))

    outdir = f'{args.bits}bit_{args.qscheme}/factors_{args.method}_seed{args.seed}/'
    os.makedirs(outdir, exist_ok=True)

    start = time.time()

    if args.method == 'admm':
        eps = 1e-8
        tol = 1e-5
        max_iter_factor = 1000
        loss_hist = []
        loss_quant_hist = []
        if run:
            wandb.config.update({"tol": tol, 
                                 "eps": eps,
                                 "max_iter": args.max_iter_als,
                                 "max_iter_factor": max_iter_factor})
        # 3-tensor case
        if weight.ndim == 3:
            A, B, C = init_factors(weight, rank=args.rank, init=args.init, 
                                   device=device, seed=args.seed)
            print(weight.dtype, A.dtype)
            U_A = torch.zeros_like(A, device=device)
            U_B = torch.zeros_like(B, device=device)
            U_C = torch.zeros_like(C, device=device)

            for i in tqdm(range(args.max_iter_als)):
                G = B.T @ B * (C.T @ C)
                # mttkrp
                F = torch.einsum('abc,cr,br->ar', weight, C, B)
                A, U_A = admm_iteration(A, U_A, F, G, 
                                        max_iter=max_iter_factor, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                A_quantized = quantize_tensor(A, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A * (C.T @ C)
                F = torch.einsum('abc,cr,ar->br', weight, C, A)
                B, U_B = admm_iteration(B, U_B, F, G, 
                                        max_iter=max_iter_factor, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                B_quantized = quantize_tensor(B, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A * (B.T @ B)
                F = torch.einsum('abc,br,ar->cr', weight, B, A)
                C, U_C = admm_iteration(C, U_C, F, G, 
                                        max_iter=max_iter_factor, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                C_quantized = quantize_tensor(C, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                error = squared_relative_diff(weight, 
                                              torch.einsum('ir,jr,kr->ijk', 
                                                           A, B, C))
                quantized_error = squared_relative_diff(weight, 
                                                        torch.einsum('ir,jr,kr->ijk', 
                                                                     A_quantized, 
                                                                     B_quantized, 
                                                                     C_quantized))
                loss_hist.append(error)
                loss_quant_hist.append(quantized_error)
                if run:
                    wandb.log({'rec_error': error, 
                               'quant_rec_error': quantized_error})
                if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < tol: 
                    break 
                # safe against exploding error
                if len(loss_hist) > 10 and loss_hist[-1] - loss_hist[-10] > 1e-3:
                    break

            factors = [A, B, C]
            factors_quantized = [A_quantized, B_quantized, C_quantized]
        # matrix case
        elif weight.ndim == 2:
            A, B = init_factors(weight, rank=args.rank, init=args.init, 
                                device=device, seed=args.seed)
            U_A = torch.zeros_like(A, device=device)
            U_B = torch.zeros_like(B, device=device)

            for i in tqdm(range(args.max_iter_als)):
                G = B.T @ B
                F = weight @ B
                A, U_A = admm_iteration(A, U_A, F, G, 
                                        max_iter=max_iter_factor, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                A_quantized = quantize_tensor(A, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A
                F = weight.T @ A
                B, U_B = admm_iteration(B, U_B, F, G, 
                                        max_iter=max_iter_factor, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                B_quantized = quantize_tensor(B, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                error = squared_relative_diff(weight, A @ B.T)
                quantized_error = squared_relative_diff(weight, A_quantized @ B_quantized.T)
                loss_hist.append(error)
                loss_quant_hist.append(quantized_error)
                if run:
                    wandb.log({'rec_error': error, 
                               'quant_rec_error': quantized_error})
                if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < tol: 
                    break 
                # safe against exploding error
                if len(loss_hist) > 10 and loss_hist[-1] - loss_hist[-10] > 1e-3:
                    break

                factors = [A, B]
                factors_quantized = [A_quantized, B_quantized]

        else: 
            raise ValueError('Incorrect number of dimentions in weight tensor')

        torch.save(loss_hist, 
          f'{outdir}/{args.layer}_{args.method}_{args.init}_rank_{args.rank}_losshist.pt')
        torch.save(loss_quant_hist, 
          f'{outdir}/{args.layer}_{args.method}_{args.init}_rank_{args.rank}_lossquanthist.pt')

    elif args.method == 'parafac':
        tl.set_backend('pytorch')
#         _, factors = parafac(weight, rank=args.rank, init=args.init, random_state=args.seed, 
#                              tol=1e-8, stop_criterion='rec_error_deviation', n_iter_max=5000)
        _, factors = parafac(weight, rank=args.rank, init=args.init, random_state=args.seed, 
                             tol=1e-8, n_iter_max=5000)
        factors_quantized = [quantize_tensor(factor, 
                                             qscheme=args.qscheme, 
                                             bits=args.bits) for factor in factors]
    elif args.method == 'parafac_epc':
        tl.set_backend('pytorch')
        _, factors = parafac_epc(weight, rank=args.rank, init=args.init,
                                 als_maxiter=500, als_tol=1e-6,
                                 epc_maxiter=500)
        factors = [torch.tensor(factor) for factor in factors]
        factors_quantized = [quantize_tensor(factor, 
                                             qscheme=args.qscheme, 
                                             bits=args.bits) for factor in factors]

    else:
        raise ValueError('Method can be one of [admm, parafac]')

    end = time.time()
    print('Factorization took {} minutes'.format((end-start)/60))

    for mode, factor in enumerate(factors):
        torch.save(factor, 
            f'{outdir}/{args.layer}_{args.method}_{args.init}_rank_{args.rank}_mode_{mode}.pt')

    if weight.ndim == 3:
        print('Factorization error is {} for usual and {} for quantized'.format(
            squared_relative_diff(weight, torch.einsum('ir,jr,kr->ijk', *factors)),
            squared_relative_diff(weight, torch.einsum('ir,jr,kr->ijk', *factors_quantized))
        ))
#     elif X.ndim == 2:
#         pass
    else:
        raise ValueError('Incorrect number of dimentions in weight tensor')

    if run: run.finish()
        
        
if __name__ == '__main__':
    main()
