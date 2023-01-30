from flopco import FlopCo

import torch
from torchvision.models import resnet18

from tqdm import tqdm
import numpy as np
import os
import random

from source.data import get_imagenet_test_loader, get_imagenet_train_val_loaders
from source.eval import accuracy
from source.admm import build_cp_layer
from source.utils import bncalibrate_model
from source.rank_map import get_rank_map

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = []
    run_name.append(f"m={args.method}")
    run_name.append(f"b={args.bits}")
    run_name.append(f"e={args.eps}")
    run_name.append(f"d={args.decomp}")
    run_name.append(f"s={args.seed}")
    run_name.append(f"{args.qscheme}")
    run_name.append(f"calibrated_{args.calibration_samples}")
    if args.no_layer1: run_name.append("no_layer1")
    
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
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        help="Number of quantization bits.")
    parser.add_argument("--calibration_samples",
                        required=True, 
                        type=int,
                        help="Number of samples for calibration.")
    parser.add_argument("--batch_size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--seed",
                        required=True, 
                        type=int,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_symmetric, tensor_affine, tensor_mse, tensor_minmaxlog]")
    parser.add_argument("--eps",
                        required=True,
                        type=float)
    parser.add_argument("--decomp",
                        required=True,
                        type=str)
    parser.add_argument("--no_layer1",
                        action="store_true",
                        help='If True, layer1.0.conv1 is not factorized.')
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Running on: {device}')
    
    args = parse_args()
    logging.info(f'Args: {args}')
    
    set_seed(args.seed)
    
    if args.with_wandb:
        import wandb
        run = wandb.init(config=args, 
                         name=run_name(args))
    else:
        run=None
        
    # original model
    model = resnet18(pretrained=True).to(device)
    model = model.to(device)
    model.eval()
    
    # datasets
    train_loader, val_loader = get_imagenet_train_val_loaders(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/',
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       val_perc=0.04,
                                       shuffle=True,
                                       random_seed=args.seed)
    test_loader = get_imagenet_test_loader(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/', 
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=False)
    
    # original model macs
    model_stats = FlopCo(model.to(device), img_size=(1, 3, 224, 224), device=device)
    orig_macs = 0
    for x in model_stats.macs.values():
        orig_macs += x[0]
        
    # loading factorized weights
    rank_map = get_rank_map(args.eps, args.decomp)
    for module in ['layer1', 'layer2', 'layer3', 'layer4']:
        for layer_path in [f'{module}.0.conv1', f'{module}.0.conv2', 
                           f'{module}.1.conv1', f'{module}.1.conv2']:
            #  layer1.0.conv1 is crucial for parafac factorization
            if layer_path == 'layer1.0.conv1': 
                if args.method == 'parafac' or args.no_layer1: continue

            lname, lidx, ltype = layer_path.split('.')
            lidx = int(lidx)
            layer = model.__getattr__(lname)[lidx].__getattr__(ltype)
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            cin = layer.in_channels
            cout = layer.out_channels
            rank = rank_map[layer_path]
            bias = layer.bias
            if bias is not None: bias = bias.detach()

            factor_name = os.path.join(f"{args.bits}bit_{args.qscheme}",
                                       f"factors_{args.method}_seed{args.seed}", 
                                       f"{layer_path}_{args.method}_random_rank_{rank}_mode_")
            logging.info(f'loading factors: {factor_name}')
            
            A = torch.load(factor_name+'0.pt').to(device)
            assert A.dtype == torch.float 
            B = torch.load(factor_name+'1.pt').to(device)
            C = torch.load(factor_name+'2.pt').to(device)

            model.__getattr__(lname)[lidx].__setattr__(
                ltype, build_cp_layer(rank, [A,B,C], bias, cin, cout, kernel_size, padding, stride).to(device))
            
    # factorized model macs
    model_stats = FlopCo(model.to(device), img_size=(1, 3, 224, 224), device=device)
    redc_macs = 0
    for x in model_stats.macs.values():
        redc_macs += x[0]
    
    logging.info(f'Macs reduced: {redc_macs / orig_macs}')
    if run: wandb.log({'macs_reduced': redc_macs / orig_macs})
        
    acc = accuracy(model, test_loader, device=device)
    logging.info(f'Factorized accuracy: {acc}')
    if run: wandb.log({'acc_factorized': acc})
        
    # batchnorm calibration
    model = bncalibrate_model(model, train_loader, 
                              num_samples=args.calibration_samples, 
                              device=device)
    
    acc = accuracy(model, test_loader, device=device)
    logging.info(f'Calibrated accuracy: {acc}')
    if run: wandb.log({'acc_calibrated': acc})
        
    # saving calibrated checkpoint
    os.makedirs('checkpoints/', exist_ok=True)
    torch.save(model, 'checkpoints/'+run_name(args))
    logging.info(f'Saved model to: checkpoints/{run_name(args)}')
    
    if run: run.finish()

        
if __name__ == '__main__':
    main()