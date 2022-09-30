import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

import pandas as pd

from data import get_imagenet_train_val_loaders, get_imagenet_test_loader
from utils_torch import *

from argparse import ArgumentParser
import os
import wandb


rank_map = {'conv1': [81, 40, 20, 10],
 'layer1.0.conv1': [269, 134, 67, 33],
 'layer1.0.conv2': [269, 134, 67, 33],
 'layer1.1.conv1': [269, 134, 67, 33],
 'layer1.1.conv2': [269, 134, 67, 33],
 'layer2.0.conv1': [366, 183, 91, 45],
 'layer2.0.conv2': [556, 278, 139, 69],
 'layer2.0.downsample': [42, 21, 10, 5],
 'layer2.1.conv1': [556, 278, 139, 69],
 'layer2.1.conv2': [556, 278, 139, 69],
 'layer3.0.conv1': [750, 375, 187, 93],
 'layer3.0.conv2': [1132, 566, 283, 141],
 'layer3.0.downsample': [85, 42, 21, 10],
 'layer3.1.conv1': [1132, 566, 283, 141],
 'layer3.1.conv2': [1132, 566, 283, 141],
 'layer4.0.conv1': [1518, 759, 379, 189],
 'layer4.0.conv2': [2283, 1141, 570, 285],
 'layer4.0.downsample': [170, 85, 42, 21],
 'layer4.1.conv1': [2283, 1141, 570, 285],
 'layer4.1.conv2': [2283, 1141, 570, 285],
 'fc': [338, 169, 84, 42]}

parser = ArgumentParser()
parser.add_argument("-m", "--method", dest="method", required=True, type=str,
                    help="admm or parafac")
parser.add_argument("-r", "--reduction", dest="reduction", required=True, type=int,
                    help="order of parameter reduction")
parser.add_argument("-s", "--seed", dest="seed", required=True, type=int)
parser.add_argument("-b", "--bits", dest="bits", required=True, type=int)
args = parser.parse_args()

bits = args.bits
print('bits:', bits)

train_loader, val_loader = get_imagenet_train_val_loaders(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/',
                                       batch_size=512,
                                       num_workers=1,
                                       pin_memory=True,
                                       val_perc=0.04,
                                       shuffle=True,
                                       random_seed=5)

test_loader = get_imagenet_test_loader(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/', 
                                       batch_size=500,
                                       num_workers=1,
                                       pin_memory=True,
                                       shuffle=False)

device = torch.device('cuda:0')
print('device:', device)

orig_model = resnet18(pretrained=True).to(device)
orig_model.eval()
orig_model_quant = resnet18(pretrained=True).to(device)
orig_model_quant.eval()

reduction = args.reduction
print('reduction:', reduction)
if reduction == 1:
    reduction_idx = 0
elif reduction == 2:
    reduction_idx = 1
elif reduction == 4:
    reduction_idx = 2
elif reduction == 8:
    reduction_idx = 3
else:
    raise ValueError('Reduction can be one of [1, 2, 4, 8]')

seed = args.seed
print('seed:', seed)

method = args.method
print('method:', method)
if method == 'admm':
    PREFIX = 'ADMM'
elif method == 'parafac':
    PREFIX = 'Parafac'
else: 
    raise ValueError('Method can be one of [admm, parafac]')

df = pd.DataFrame([], columns=['Layer Name', 'Reduction', 'Seed', 
                               f'{PREFIX} Acc', f'{PREFIX}+TQuant Acc', f'{PREFIX}+TQuant+Calib Acc'])
df.loc[0] = [None] * len(df.columns)
df.loc[0]['Reduction'] = reduction
df.loc[0]['Seed'] = seed
    
run = wandb.init(config=args, 
                 project=f"resnet18-factquant-evalmult-{bits}bit-symmetric", 
                 entity="darayavaus",
                 name=f'{method}_all_{reduction}x_seed{seed}')
wandb.config.update({"train_dataset_seed": 5})

if method == 'parafac':
    wandb.config.update({"tol": 1e-8, 
                         "stop_criterion": 'rec_error_deviation',
                         "n_iter_max": 5000})

# conv1
# layer_path = 'conv1'
# layer = orig_model.__getattr__(layer_path)
# kernel_size = layer.kernel_size
# stride = layer.stride
# padding = layer.padding
# cin = layer.in_channels
# cout = layer.out_channels
# rank = rank_map[layer_path][reduction_idx]
# bias = layer.bias
# if bias is not None: bias = bias.detach()
# print(f'loading factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt')
# A = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt').to(device)
# quantized_A = quantize_tensor(A, qscheme=torch.per_tensor_symmetric, bits=bits)
# B = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_1.pt').to(device)
# quantized_B = quantize_tensor(B, qscheme=torch.per_tensor_symmetric, bits=bits)
# C = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_2.pt').to(device)
# quantized_C = quantize_tensor(C, qscheme=torch.per_tensor_symmetric, bits=bits)
# assert A.dtype == torch.float and quantized_A.dtype == torch.float
# orig_model.__setattr__(layer_path, build_cp_layer(rank, [A,B,C], bias, cin, cout, kernel_size, padding, stride).to(device))
# orig_model_quant.__setattr__(layer_path, 
#                              build_cp_layer(rank, [quantized_A, quantized_B, quantized_C], bias, cin, cout, kernel_size, padding, stride).to(device))
X = orig_model.conv1.weight.detach().to(device)
with torch.no_grad():
    orig_model_quant.conv1.weight = nn.Parameter(quantize_tensor(X, qscheme=torch.per_tensor_symmetric, bits=bits), requires_grad=True)

# layer1.* - layer4.*
for module in ['layer1', 'layer2', 'layer3', 'layer4']:
    for layer_path in [f'{module}.0.conv1', f'{module}.0.conv2', 
#                        f'{module}.0.downsample',
                       f'{module}.1.conv1', f'{module}.1.conv2']:
        # there is no layer1.0.downsample layer
        if layer_path == 'layer1.0.downsample': continue
        lname, lidx, ltype = layer_path.split('.')
        lidx = int(lidx)
        layer = orig_model.__getattr__(lname)[lidx].__getattr__(ltype)
        if ltype == 'downsample': layer = layer[0]
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        cin = layer.in_channels
        cout = layer.out_channels
        rank = rank_map[layer_path][reduction_idx]
        bias = layer.bias
        if bias is not None: bias = bias.detach()
        
        print(f'loading factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt')
        A = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt').to(device)
        quantized_A = quantize_tensor(A, qscheme=torch.per_tensor_symmetric, bits=bits)
        B = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_1.pt').to(device)
        quantized_B = quantize_tensor(B, qscheme=torch.per_tensor_symmetric, bits=bits)
        assert A.dtype == torch.float and quantized_A.dtype == torch.float
        if ltype == 'downsample':
#             orig_model.__getattr__(lname)[lidx].__getattr__(ltype)[0] = build_cp2conv_layer(rank, [A,B], bias, cin, cout, 
#                                                                                             padding, stride).to(device)
#             orig_model_quant.__getattr__(lname)[lidx].__getattr__(ltype)[0] = build_cp2conv_layer(rank, [quantized_A, quantized_B], 
#                                                                                             bias, cin, cout, padding, stride).to(device)
            X = orig_model.__getattr__(lname)[lidx].__getattr__(ltype)[0].weight.detach().to(device)
            orig_model_quant.__getattr__(lname)[lidx].__getattr__(ltype)[0].weight = \
                nn.Parameter(quantize_tensor(X, qscheme=torch.per_tensor_symmetric, bits=bits), requires_grad=True)
        else:
            C = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_2.pt').to(device)
            quantized_C = quantize_tensor(C, qscheme=torch.per_tensor_symmetric, bits=bits)
            
            orig_model.__getattr__(lname)[lidx].__setattr__(
                ltype, build_cp_layer(rank, [A,B,C], bias, cin, cout, kernel_size, padding, stride).to(device))
            orig_model_quant.__getattr__(lname)[lidx].__setattr__(
                ltype, build_cp_layer(rank, [quantized_A, quantized_B, quantized_C], bias, cin, cout, kernel_size, padding, stride).to(device))
            
#fc
# layer_path = 'fc'
# layer = orig_model.__getattr__(layer_path)
# bias = layer.bias
# if bias is not None: bias = bias.detach()
# fin = layer.in_features
# fout = layer.out_features
# rank = rank_map[layer_path][reduction_idx]
# print(f'loading factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt')
# A = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_0.pt').to(device)
# quantized_A = quantize_tensor(A, qscheme=torch.per_tensor_symmetric, bits=bits)
# B = torch.load(f'{bits}bit_symmetric/factors_{method}_seed{seed}/{layer_path}_{method}_random_rank_{rank}_mode_1.pt').to(device)
# quantized_B = quantize_tensor(B, qscheme=torch.per_tensor_symmetric, bits=bits)
# assert A.dtype == torch.float and quantized_A.dtype == torch.float
# orig_model.__setattr__('fc', build_cpfc_layer(rank, [A,B], bias, fin, fout))
X = orig_model.fc.weight.detach().to(device)
orig_model_quant.fc.weigth = nn.Parameter(quantize_tensor(X, qscheme=torch.per_tensor_symmetric, bits=bits), requires_grad=True)

acc = accuracy(orig_model, test_loader, device=device, num_classes=1000)
df.loc[0][f'{PREFIX} Acc'] = acc
wandb.log({f'{PREFIX} Acc': acc})

acc = accuracy(orig_model_quant, test_loader, device=device, num_classes=1000)
df.loc[0][f'{PREFIX}+TQuant Acc'] = acc
wandb.log({f'{PREFIX}+TQuant Acc': acc})

orig_model_quant = calibrate(orig_model_quant.to(device), train_loader, device=device, num_batches=100)
acc = accuracy(orig_model_quant.to(device), test_loader, device=device, num_classes=1000)
df.loc[0][f'{PREFIX}+TQuant+Calib Acc'] = acc
wandb.log({f'{PREFIX}+TQuant+Calib Acc': acc})

csv_path = f'{bits}bit_symmetric/csv_all_{method}_seed{args.seed}_intdownfcandconv1/all_reduction{reduction}.csv'
if not os.path.exists('/'.join(csv_path.split('/')[0:2])):
    os.mkdir('/'.join(csv_path.split('/')[0:2]))
df.to_csv(csv_path, index=False)
wandb.log({"table": df})

model_path = f'{bits}bit_symmetric/models_all_{method}_seed{seed}_intdownfcandconv1/all_{method}_reduction{reduction}'
if not os.path.exists('/'.join(model_path.split('/')[0:2])):
    os.mkdir('/'.join(model_path.split('/')[0:2]))
torch.save(orig_model_quant.state_dict(), model_path)
wandb.config.update({"model_path": model_path})

run.finish()
