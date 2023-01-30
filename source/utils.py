import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.
    
    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    
    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def get_layer_by_name(model, mname):
    '''
    Extract layer using layer name
    '''
    module = model
    mname_list = mname.split('.')
    for mname in mname_list:
        module = module._modules[mname]

    return module


def bncalibrate_layer(model, train_loader, n_batches = 200000//256, 
                           layer_name = None, device="cuda:0"):
    '''
    Update batchnorm statistics for layers after layer_name
    Parameters:
    model                   -   Pytorch model
    train_loader            -   Training dataset dataloader, Pytorch Dataloader
    n_callibration_batches  -   Number of batchnorm callibration iterations, int
    layer_name              -   Name of layer after which to update BN statistics, string or None
                                (if None updates statistics for all BN layers)
    device                  -   Device to store the model, string
    '''
    
    # switch batchnorms into the mode, in which its statistics are updated
    model.to(device).eval() 
    layer_passed = False
    
    if layer_name is not None:
        #freeze batchnorms before replaced layer
        for lname, l in model.named_modules():

            if lname == layer_name:
                layer_passed = True
            
            if (isinstance(l, nn.BatchNorm2d)) and layer_passed:
                if layer_passed:
                    l.train()
                else:
                    l.eval()

    with torch.no_grad():            

        for i, (data, _) in enumerate(train_loader):
            _ = model(data.to(device))

            if i > n_batches:
                break
            
        del data
        torch.cuda.empty_cache()
        
    model.eval()
    return model


def bncalibrate_model(model, dataset_loader, num_samples=1000, device='cuda'):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Set BN to train regime
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    count = 0
    for (x, _) in tqdm(dataset_loader):
        if count > num_samples:
            break

        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
            
        count += dataset_loader.batch_size

    return model
