from flopco import FlopCo

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from tqdm import tqdm
import numpy as np
import os
import random

from source.data import get_imagenet_test_loader, get_imagenet_train_val_loaders
from source.eval import accuracy
from source.models import build_cp_layer, build_cp2conv_layer, ResNet18Quant, regnet_y_3_2gf
from source.utils import bncalibrate_model
from source.rank_map import get_rank_map
from source.layer_map import get_layer_list

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = [args.model_name]
    run_name.append(f"m={args.method}")
    run_name.append(f"b={args.bits}")
    run_name.append(f"r={args.reduction_rate}")
    run_name.append(f"i={args.init}")
    run_name.append(f"s={args.seed}")
    run_name.append(f"{args.qscheme}")
    run_name.append(f"calibrated_{args.calibration_samples}")
    
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--model-name",
                        type=str,
                        required=True,
                        help="[resnet18, resnet50]")
    parser.add_argument("--data-root",
                        type=str,
                        help="Root dir of ImageNet Dataset")
    parser.add_argument("--method", 
                        type=str, 
                        required=True,
                        help="[admm, parafac, parafac-epc]")
    parser.add_argument("--init",
                        type=str,
                        required=False,
                        default='random',
                        help="Method to initialize factors. [random, parafac-epc]")
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        help="Number of quantization bits.")
    parser.add_argument("--calibration-samples",
                        required=False,
                        default=2048,
                        type=int,
                        help="Number of samples for BatchNorm calibration.")
    parser.add_argument("--batch-size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--seed",
                        type=int,
                        required=False, 
                        default=42,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_mseminmax_symmetric, tensor_minmax]")
    parser.add_argument("--reduction-rate",
                        required=False,
                        type=float)
    parser.add_argument("--num-workers",
                        type=int,
                        required=False,
                        default=4)
    
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
    if args.model_name == 'resnet18':
        # weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
        # model = ResNet18Quant(num_classes=len(weights.meta["categories"]))
        # model.load_state_dict(weights.get_state_dict(progress=True))
        model = resnet18(pretrained=True)
    elif args.model_name == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        raise ValueError(f"Unrecognized model name: {args.model_name}")
    model = model.to(device)
    model.eval()
    
    # datasets
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
    
    # original model macs
    model_stats = FlopCo(model.to(device), 
                         img_size=(1, 3, 224, 224), 
                         instances=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d],
                         device=device)
    orig_macs = 0
    for x in model_stats.macs.values():
        orig_macs += x[0]
        
    # loading factorized weights
    rank_map = get_rank_map(args.model_name, rate=args.reduction_rate)
    layer_list = get_layer_list(args.model_name, 
                                downsample=False, 
                                conv1=False)
    for layer_path in layer_list:
        layer = model 
        for attr in layer_path.split('.'):
            layer = layer.__getattr__(attr)
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        cin = layer.in_channels
        cout = layer.out_channels
        groups = layer.groups
        rank = rank_map[layer_path]
        bias = layer.bias
        if bias is not None: bias = bias.detach()

        factor_name = os.path.join(f"{args.bits}bit_{args.qscheme}",
                                   f"factors_{args.method}_seed{args.seed}", 
                                   f"{layer_path}_{args.method}_{args.init}_rank_{rank}_")
        logging.info(f'loading factors: {factor_name}')

        A = torch.load(factor_name+'mode_0.pt').to(device)
        assert A.dtype == torch.float 
        B = torch.load(factor_name+'mode_1.pt').to(device)
        # kernel_size == (1,1) => equivalent to a Linear layer
        if kernel_size != (1, 1):
            C = torch.load(factor_name+'mode_2.pt').to(device)
            factorized_layer = build_cp_layer(rank, [A,B,C], bias, cin, cout, 
                                              kernel_size, padding, stride, groups).to(device)
        else:
            factorized_layer = build_cp2conv_layer(rank, [A,B], bias, cin, cout, 
                                                   padding, stride).to(device)

        layer = model
        for attr in layer_path.split('.')[:-1]:
            layer = layer.__getattr__(attr)
        layer.__setattr__(layer_path.split('.')[-1], factorized_layer)
            
    # factorized model macs
    model_stats = FlopCo(model.to(device), 
                         img_size=(1, 3, 224, 224), 
                         instances=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d],
                         device=device)
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
