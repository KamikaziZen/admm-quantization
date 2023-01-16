import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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


def estimate_macs(model, layer_name, rank, device):
    """Returns original and reduced macs based on reduction rank
    original macs = C_i * W_k * H_k * C_o * W_o * H_o
    reduced macs = rank * C_i * W_i * H_i + rank * W_k * H_k * W_o * H_o + rank * C_o * W_o * H_o
    where:
        C_i - number of input channels
        C_o - number of output channels
        W_o - width of the output image
        H_o - height of the output image
        W_i - width of the input image
        H_i - height of the input image
        W_k - width of the kernel
        H_k - height of the kernel
    """
    input_shape = output_shape = (1, 3, 224, 224)
    layer = None
    x = torch.rand(*input_shape).to(device)
    model.eval()
    with torch.no_grad():
        for lname, layer in model.named_modules():
            if not (isinstance(layer, nn.Conv2d) 
                    or isinstance(layer, nn.BatchNorm2d) 
                    or isinstance(layer, nn.MaxPool2d) 
                    or isinstance(layer, nn.ReLU)): continue
            if 'downsample' in lname: continue
            input_shape = x.shape
            x = layer(x)
            output_shape = x.shape
            if lname == layer_name: break
                
    if not isinstance(layer, nn.Conv2d):
        raise NotImplementedError('Function estimate_macs works only for Conv2d layers')
        
    orig_macs = layer.in_channels * layer.kernel_size[-1] * layer.kernel_size[-2] \
                * layer.out_channels * output_shape[-1] * output_shape[-2]
    redc_macs = rank * layer.in_channels * input_shape[-1] * input_shape[-2] \
                + rank * layer.kernel_size[-1] * layer.kernel_size[-2] \
                  * output_shape[-1] * output_shape[-2] \
                + rank * layer.out_channels * output_shape[-1] * output_shape[-2]
    
    return orig_macs, redc_macs