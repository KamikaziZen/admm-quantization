import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
import pandas as pd

from data import get_imagenet_train_val_loaders, get_imagenet_test_loader
from utils_torch import *

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-l", "--layer", dest="layer", required=True, type=str,
                    help="name of a layer to evaluate(for example, layer2.1.conv1 means model.layer2[1].conv1)")
parser.add_argument("-r", "--rank", dest="rank", required=True, type=int,
                    help="rank of decomposition")
args = parser.parse_args()


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


orig_model = resnet18(pretrained=True)
orig_model.eval()

lname, lidx, ltype = args.layer.split('.')
lidx = int(lidx)
layer = orig_model.__getattr__(lname)[lidx].__getattr__(ltype)
print('layer:', layer)

device = torch.device('cuda:0')
print('device:', device)

rank = args.rank
print('rank:', rank)

df = pd.DataFrame([], columns=['Layer Name', 'Rank', 'Parafac RecErr', 'Parafac TQuant RecErr', 'Parafac Acc',
       'Parafac+TQuant Acc', 'Parafac+TQuant+Calib Acc', 'ADMM RecErr', 'ADMM TQuant RecErr',
       'ADMM Acc', 'ADMM+TQuant Acc', 'ADMM+TQuant+Calib Acc'])
df.loc[0] = [None] * len(df.columns)
df.loc[0]['Layer Name'] = args.layer
df.loc[0]['Rank'] = rank

kernel_size = layer.kernel_size
stride = layer.stride
padding = layer.padding
cin = layer.in_channels
cout = layer.out_channels

X = layer.weight.detach()
X = X.reshape((X.shape[0], X.shape[1], -1))
bias = layer.bias
if bias is not None: bias = bias.detach()

scale = (X.max() - X.min()) / scale_denom
zero_point = (-q - (X.min() / scale).int()).int()
zero_point = torch.clamp(zero_point, -q, q - 1)
    
A = torch.load(f'factors/{args.layer}_parafac_random_errdev_default_rank_{rank}_mode_0.pt')
B = torch.load(f'factors/{args.layer}_parafac_random_errdev_default_rank_{rank}_mode_1.pt')
C = torch.load(f'factors/{args.layer}_parafac_random_errdev_default_rank_{rank}_mode_2.pt')
quantized_A = quantize_tensor(A, scale=scale, zero_point=zero_point)
quantized_B = quantize_tensor(B, scale=scale, zero_point=zero_point)
quantized_C = quantize_tensor(C, scale=scale, zero_point=zero_point)
assert A.dtype == torch.float and quantized_A.dtype == torch.float

error = squared_relative_diff(X, np.einsum('ir,jr,kr->ijk', A, B, C))
quantized_error = squared_relative_diff(X, np.einsum('ir,jr,kr->ijk', quantized_A, quantized_B, quantized_C))
df.loc[0]['Parafac RecErr'] = error
df.loc[0]['Parafac TQuant RecErr'] = quantized_error

# orig_model = resnet18(pretrained=True)
# orig_model.eval()
# orig_model.layer2[1].conv1 = build_cp_layer(rank, [A,B,C], bias, cin, cout, kernel_size, padding, stride)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['Parafac Acc'] = acc

# orig_model = resnet18(pretrained=True)
# orig_model.eval()
# orig_model.layer2[1].conv1 = build_cp_layer(rank, [quantized_A, quantized_B, quantized_C], bias, cin, cout, kernel_size, padding, stride)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['Parafac+TQuant Acc'] = acc

# orig_model = calibrate(orig_model.to(device), train_loader, device=device, num_batches=100)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['Parafac+TQuant+Calib Acc'] = acc

A = torch.load(f'factors_proj/{args.layer}_admm_random_rank_{rank}_mode_0.pt')
B = torch.load(f'factors_proj/{args.layer}_admm_random_rank_{rank}_mode_1.pt')
C = torch.load(f'factors_proj/{args.layer}_admm_random_rank_{rank}_mode_2.pt')
quantized_A = quantize_tensor(A, scale=scale, zero_point=zero_point)
quantized_B = quantize_tensor(B, scale=scale, zero_point=zero_point)
quantized_C = quantize_tensor(C, scale=scale, zero_point=zero_point)
assert A.dtype == torch.float and quantized_A.dtype == torch.float

error = squared_relative_diff(X, np.einsum('ir,jr,kr->ijk', A, B, C))
quantized_error = squared_relative_diff(X, np.einsum('ir,jr,kr->ijk', quantized_A, quantized_B, quantized_C))
df.loc[0]['ADMM RecErr'] = error
df.loc[0]['ADMM TQuant RecErr'] = quantized_error

# orig_model = resnet18(pretrained=True)
# orig_model.eval()
# orig_model.layer2[1].conv1 = build_cp_layer(rank, [A,B,C], bias, cin, cout, kernel_size, padding, stride)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['ADMM Acc'] = acc

# orig_model = resnet18(pretrained=True)
# orig_model.eval()
# orig_model.layer2[1].conv1 = build_cp_layer(rank, [quantized_A, quantized_B, quantized_C], 
#                                             bias, cin, cout, kernel_size, padding, stride)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['ADMM+TQuant Acc'] = acc

# orig_model = calibrate(orig_model.to(device), train_loader, device=device, num_batches=100)
# acc = accuracy(orig_model.to(device), test_loader, device='cuda', num_classes=1000)
# df.loc[0]['ADMM+TQuant+Calib Acc'] = acc

df.to_csv(f'csv_proj/{args.layer}_{rank}.csv', index=False)