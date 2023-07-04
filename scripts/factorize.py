import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
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
    parser.add_argument("--model-name",
                        type=str,
                        required=True,
                        help="[resnet18, resnet50, unet]")
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--method", 
                        type=str, 
                        required=True,
                        help="[admm, parafac, parafac-epc]")
    parser.add_argument("--init",
                        type=str,
                        required=False,
                        default='random',
                        help="[random, parafac-epc]")
    parser.add_argument("--layer", 
                        type=str, 
                        required=True,
                        help="Name of a layer to decompose(for example, layer2.1.conv1 means model.layer2[1].conv1)")
    parser.add_argument("--rank", 
                        type=int, 
                        required=False,
                        help="Rank for decomposition.")
    parser.add_argument("--reduction-rate",
                        type=float,
                        required=False,
                        help="Rank is computed such that number of parameters reduce <reduction_rate> times(if not specified in arguments).")
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        help="Number of quantization bits.")
    parser.add_argument("--max_iter_als",
                        required=False,
                        default=5000, 
                        type=int,
                        help="Maximum number of ALS iterations. Used for ADMM, Parafac and Parafac-epc.")
    parser.add_argument("--max_iter_admm",
                        required=False,
                        default=1000,
                        type=int,
                        help="Maximum number of iterations of ADMM for each step of ALS.")
    parser.add_argument("--max_iter_epc",
                        required=False,
                        default=5000,
                        type=int,
                        help="Maximum number of iterations of Parafac-epc for each step of ALS. Used if method==parafac-epc.")
    parser.add_argument("--seed",
                        required=True, 
                        type=int,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_mseminmax_symmetric, tensor_minmax]")

    args = parser.parse_args()
    # Args Check
    if args.rank is None and args.reduction_rate is None:
        raise ValueError('One of [--rank, --reduction-rate] arguments must be specified.')
    if args.method not in ['admm', 'parafac', 'parafac-epc']:
        raise ValueError('Method must be on of [admm, parafac, parafac-epc].')
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on:', device)
    
    args = parse_args()
    print('Args:', args)
    
    # set all random seeds
    set_seed(args.seed)
        
    # load original unfactorized model
    if args.model_name == 'resnet18':
        model = resnet18(pretrained=True)
    elif args.model_name == 'resnet50':
        model = resnet50(pretrained=True)
    elif args.model_name == 'deit':
        state_dict = torch.load('deit.sd')
    elif args.model_name == 'unet':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    else:
        raise ValueError(f"unrecognized model name: {args.model_name}")
    # model.eval()
    
    # get weight tensor
    # layer = model 
    # for attr in args.layer.split('.'):
    #     layer = layer.__getattr__(attr)
    # weight = layer.weight.detach().to(device)
    # bias = layer.bias
    weight = state_dict[args.layer+'.weight']
    bias = state_dict.get(args.layer+'.bias')
    # kernel_size = layer.kernel_size
    # groups = layer.groups
    if bias is not None: bias = bias.detach().to(device)
    # if isinstance(layer, nn.Conv2d):
    #     # kernel_size == (1,1) => equivalent to a Linear layer
    #     if kernel_size == (1, 1):
    #         weight = weight.reshape((weight.shape[0], weight.shape[1]))
    #     else:
    #         weight = weight.reshape((weight.shape[0], weight.shape[1], -1))
    # else:
    #     raise NotImplementedError(layer)
        
    if weight.ndim == 3:
        ein_op = 'ir,jr,kr->ijk'
    elif weight.ndim == 2:
        ein_op = 'ir,jr->ij'
    else:
        raise ValueError('Incorrect number of dimentions in weight tensor')

    # if rank is not specified, compute it from reduction rate
    if args.rank is None:
        args.rank = int(weight.numel() / sum(list(weight.shape)) / args.reduction_rate)
        # if groups != 1: 
        #     while args.rank % groups != 0:
        #         args.rank += 1
            
    # create output dir to store factors
    outdir = f'{args.bits}bit_{args.qscheme}/factors_{args.method}_seed{args.seed}'
    os.makedirs(outdir, exist_ok=True)
    fileprefix = f'{args.layer}_{args.method}_{args.init}_rank_{args.rank}'
        
    # log to wandb
    if args.with_wandb:
        import wandb
        run = wandb.init(config=args, 
                         name=run_name(args))
    else:
        run=None
    
    # factorize
    start = time.time()
    if args.method == 'admm':
        eps = 1e-8
        tol = 1e-5
        loss_hist = []
        loss_quant_hist = []
        if run:
            wandb.config.update({"tol": tol, 
                                 "eps": eps})
            
        # initialization of factors
        factors = init_factors(weight, rank=args.rank, init=args.init, 
                           device=device, seed=args.seed)
            
        # if not random initialization -> save starting point in history
        if args.init != 'random':
            factors_quantized = [quantize_tensor(factor,
                                             qscheme=args.qscheme,
                                             bits=args.bits) for factor in factors]
            error = squared_relative_diff(weight,
                                          torch.einsum(ein_op, *factors))
            quantized_error = squared_relative_diff(weight,
                                                    torch.einsum(ein_op, *factors_quantized))
            loss_hist.append(error)
            loss_quant_hist.append(quantized_error)    
            if run:
                wandb.log({'rec_error': error,
                           'quant_rec_error': quantized_error})

        # 3-tensor case
        if weight.ndim == 3:
            
            A, B, C = factors
            U_A = torch.zeros_like(A, device=device)
            U_B = torch.zeros_like(B, device=device)
            U_C = torch.zeros_like(C, device=device)

            for i in tqdm(range(args.max_iter_als)):
                G = B.T @ B * (C.T @ C)
                # mttkrp
                F = torch.einsum('abc,cr,br->ar', weight, C, B)
                A, U_A = admm_iteration(A, U_A, F, G, 
                                        max_iter=args.max_iter_admm, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                A_quantized = quantize_tensor(A, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A * (C.T @ C)
                F = torch.einsum('abc,cr,ar->br', weight, C, A)
                B, U_B = admm_iteration(B, U_B, F, G, 
                                        max_iter=args.max_iter_admm, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                B_quantized = quantize_tensor(B, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A * (B.T @ B)
                F = torch.einsum('abc,br,ar->cr', weight, B, A)
                C, U_C = admm_iteration(C, U_C, F, G, 
                                        max_iter=args.max_iter_admm, 
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
                if len(loss_hist) > 10 and loss_hist[-1] - loss_hist[-5] > 1e-3:
                    break

            factors = [A, B, C]
            factors_quantized = [A_quantized, B_quantized, C_quantized]
            
        # matrix case
        elif weight.ndim == 2:
            
            A, B = factors
            U_A = torch.zeros_like(A, device=device)
            U_B = torch.zeros_like(B, device=device)

            for i in tqdm(range(args.max_iter_als)):
                G = B.T @ B
                F = weight @ B
                A, U_A = admm_iteration(A, U_A, F, G, 
                                        max_iter=args.max_iter_admm, 
                                        eps=eps, bits=args.bits, 
                                        qscheme=args.qscheme)
                A_quantized = quantize_tensor(A, 
                                              qscheme=args.qscheme, 
                                              bits=args.bits)

                G = A.T @ A
                F = weight.T @ A
                B, U_B = admm_iteration(B, U_B, F, G, 
                                        max_iter=args.max_iter_admm, 
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
                   os.path.join(outdir, fileprefix + '_losshist.pt')) 
        torch.save(loss_quant_hist, 
                   os.path.join(outdir, fileprefix + '_lossquanthist.pt'))

    elif args.method == 'parafac':
        tl.set_backend('pytorch')
#         _, factors = parafac(weight, rank=rank, init=args.init, random_state=args.seed, 
#                              tol=1e-8, stop_criterion='rec_error_deviation', n_iter_max=5000)
        _, factors = parafac(weight, rank=args.rank, init=args.init, random_state=args.seed, 
                             tol=1e-8, n_iter_max=args.max_iter_als)
        factors_quantized = [quantize_tensor(factor, 
                                             qscheme=args.qscheme, 
                                             bits=args.bits) for factor in factors]
    elif args.method == 'parafac-epc':
        tl.set_backend('pytorch')
        _, factors = parafac_epc(weight, rank=args.rank, init=args.init,
                                 als_maxiter=args.max_iter_als,
                                 epc_maxiter=args.max_iter_epc)
        factors = [torch.tensor(factor, dtype=torch.float) for factor in factors]
        factors_quantized = [quantize_tensor(factor, 
                                             qscheme=args.qscheme, 
                                             bits=args.bits) for factor in factors]

    else:
        raise ValueError('Method can be one of [admm, parafac]')

    end = time.time()
    print('Factorization took {} minutes'.format((end-start)/60))

    for mode, factor in enumerate(factors):
        torch.save(factor, 
                   os.path.join(outdir, fileprefix + f'_mode_{mode}.pt'))
    
    error = squared_relative_diff(weight, torch.einsum(ein_op, *factors))
    quantized_error = squared_relative_diff(weight, torch.einsum(ein_op, *factors_quantized))
    print('Factorization error is {} for usual and {} for quantized'.format(
        error, quantized_error))

    if run: 
        wandb.log({'rec_error': error,
                   'quant_rec_error': quantized_error})
        run.finish()
        
        
if __name__ == '__main__':
    main()
