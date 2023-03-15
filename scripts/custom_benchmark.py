from flopco import FlopCo

import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from tqdm import tqdm
import numpy as np
import os
import random
from collections import OrderedDict
import math
from flopco import FlopCo

from source.data import get_imagenet_test_loader, get_imagenet_train_val_loaders
from source.eval import accuracy, accuracy_top1top5
from source.models import build_cp_layer, ResNet18Quant
from source.utils import bncalibrate_model
from source.rank_map import get_rank_map
from source.quantization import quantize_tensor, duplicate_resnet_factorized_with_quant

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = [args.model_name]
    run_name.append(f"s={args.seed}")
    run_name.append(f"w{args.param_bw}a{args.output_bw}")
    run_name.append(f"p={args.param_qscheme}")
    run_name.append(f"o={args.output_qscheme}")
    run_name.append
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-name",
                        type=str,
                        required=False,
                        help="[resnet18]")
    parser.add_argument("--model-path",
                        required=False)
    parser.add_argument("--data-root",
                        type=str,
                        help="Root dir of ImageNet Dataset")
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--output-bw",
                        required=True, 
                        type=int,
                        help="Activations bitwidth.")
    parser.add_argument("--param-bw",
                        required=True, 
                        type=int,
                        help="Weights bitwidth.")
    parser.add_argument("--batch-size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--reduction-rate",
                        required=True, 
                        type=float)
    parser.add_argument("--param-qscheme",
                        required=True,
                        type=str)
    parser.add_argument("--output-qscheme",
                        required=True,
                        type=str)
    parser.add_argument("--observer-samples",
                        required=False,
                        default=500,
                        type=int)
    parser.add_argument("--seed",
                        required=True, 
                        type=int,
                        help="Random seed.")
    parser.add_argument("--num-workers",
                        type=int,
                        required=False,
                        default=4)
    parser.add_argument("--bits",
                        type=int,
                        required=False)
    
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
        
    # loading datasets
    train_loader, val_loader = get_imagenet_train_val_loaders(data_root=args.data_root,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       val_perc=0.04,
                                       shuffle=True,
                                       random_seed=args.seed)
    test_loader = get_imagenet_test_loader(data_root=args.data_root, 
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       shuffle=False)
    
    # loading models
    if args.model_path is not None:
        model = torch.load(args.model_path)
    elif args.model_name == 'resnet18':
        
        # randomly initialized model
        weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
        model = ResNet18Quant(num_classes=len(weights.meta["categories"]))
        model.load_state_dict(weights.get_state_dict(progress=True))
    else:
        raise ValueError('Unrecognized model name')
    model = model.to(device)
    model.eval()
    
    # counting macs and bops(macs * weight bitwidth * input bitwidth)
    model_stats = FlopCo(model, img_size=(1, 3, 224, 224), device=device,
                     instances=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d])
    macs = 0
    for x in model_stats.macs.values():
        macs += x[0]
    bops_stats = OrderedDict()
    bops = 0
    for name, macs_list in model_stats.macs.items():
        if 'bn' in name or \
            'downsample' in name or \
            name.startswith('conv1') or \
            name.startswith('fc'):
            bits = 8
        else:
            bits = args.param_bw
        bops_stats[name]  = macs_list[0] * args.output_bw * bits
        bops += bops_stats[name]
    logging.info(str(bops_stats))
    logging.info(f'Macs: {macs}, Bops: {bops}')
    if run:
        wandb.log({'macs': macs, 'bops': bops})
    
    acc1, acc5 = accuracy_top1top5(model, test_loader)
    logging.info(f'Acc1 calibrated: {acc1}, Acc5 calibrated: {acc5}')
    if run:
        wandb.log({'acc1_calibrated': acc1,
                   'acc5_calibrated': acc5})
    
        
    # quantizing weights(all except batchnorms)
    # batchnorms either have 'bn' or '0.downsample.1' as a substring
    state_dict_quant = OrderedDict()
    for name, w in model.state_dict().items():
        if 'bn' in name or '0.downsample.1' in name:
            bits = 32
        elif 'downsample' in name or \
            name.startswith('conv1') or \
            name.startswith('fc'):
            bits = 8
        else: 
            bits = args.param_bw
        print(f'Param: {name}, Bits: {bits}')

        if bits == 32:
            state_dict_quant[name] = w
        else: 
            state_dict_quant[name] = quantize_tensor(w, qscheme=args.param_qscheme, bits=bits, num_attempts=1000)
    model.load_state_dict(state_dict_quant)
    model.eval()
    
    acc1, acc5 = accuracy_top1top5(model, test_loader)
    logging.info(f'Acc1 after weight quantization(BN32): {acc1}, Acc5 after weight quantization(BN32): {acc5}')
    if run:
        wandb.log({'acc1_wquantized(BN32)': acc1,
                   'acc5_wquantized(BN32)': acc5})
        
    # calibrating batchnorms after weight quantization:
    model = bncalibrate_model(model, train_loader, 
                              num_samples=2048, 
                              device=device)
    
    # quantizing batchnorms to 8 bits
    state_dict_quant = OrderedDict()
    for name, w in model.state_dict().items():
        if 'bn' in name or '0.downsample.1' in name:
            state_dict_quant[name] = quantize_tensor(w, qscheme=args.param_qscheme, bits=8, num_attempts=1000)
        else:
            state_dict_quant[name] = w
    model.load_state_dict(state_dict_quant)
    model.eval()
    
    acc1, acc5 = accuracy_top1top5(model, test_loader)
    logging.info(f'Acc1 after weight quantization: {acc1}, Acc5 after weight quantization: {acc5}')
    if run:
        wandb.log({'acc1_wquantized': acc1,
                   'acc5_wquantized': acc5})
    
        
    # adding activation observers/quantizers
    model = duplicate_resnet_factorized_with_quant(model, ltype=args.output_qscheme, bits=args.output_bw, 
                                                   counter=args.observer_samples)
    logging.info(str(model))
    # observing (calibrating activations quantization parameters)
    # observers calibrate parameters until args.observer_samples have passed
    # then they turn to quantization regime
    accuracy_top1top5(model, val_loader, n_sample=args.observer_samples)
    
    # evaluating
    acc1, acc5 = accuracy_top1top5(model, test_loader)
    logging.info(f'Acc1 after activations quantization: {acc1}, Acc5 after activations quantization: {acc5}')
    if run:
        wandb.log({'acc1_aquantized': acc1,
                   'acc5_aquantized': acc5})
    
    if run: run.finish()
        
        
if __name__ == '__main__':
    main()
