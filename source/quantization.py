import torch


def quantize_tensor_mse(x, bits, qscheme=None, scale=None):
    def _quantize(x_float, maxval, per_channel_axis=0):
        mantissa_bits = torch.tensor(3)
        exponent_bits = 7 - mantissa_bits
        
        if maxval.shape and maxval.shape[0] != 1 and len(maxval.shape) != len(x_float.shape):
            new_shape = [1] * len(x_float.shape)
            new_shape[per_channel_axis] = -1
            maxval = maxval.view(new_shape)
        
        bias = 2 ** exponent_bits - torch.log2(maxval) + torch.log2(2 - 2 ** (-mantissa_bits)) - 1
        x_clipped = torch.min(torch.max(x_float, -maxval), maxval)

        log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()
        log_scales = torch.clamp(log_scales, 1.)

        scales = 2. ** (log_scales - mantissa_bits - bias)

        return torch.round(x_clipped / scales) * scales
    
    mx = torch.max(torch.abs(x.min()), torch.abs(x.max()))
    lsp = [torch.linspace(0.1 * mx.item(), 1.2 * mx.item(), 111)]
    # 111 x 1
    linspaces = torch.stack(lsp).to(x.device).transpose(0, 1)
    
    mses = torch.zeros_like(linspaces)
    
    meandims = list(torch.arange(len(x.shape)))
    
    for i, maxval in enumerate(linspaces):
        xfp = _quantize(x, maxval)
        mse = ((x - xfp) ** 2).mean(dim=meandims)
        mses[i, :] = mse

    best_mse = mses.argmin(0)
    maxval = torch.tensor([linspaces[best_mse[i], i] for i in range(linspaces.shape[-1])]).to(x.device)

    return _quantize(x, maxval)


def quantize_tensor(x, bits, qscheme):
    q = 2 ** (bits-1)
    scale_denom = 2*q - 1
        
    xmin = x.min()
    xmax = x.max()
        
    if qscheme == 'tensor_symmetric':
        if scale is None:
            scale = 2 * torch.where(xmin.abs() > xmax, xmin.abs(), xmax) / scale_denom

        return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale
    
    elif qscheme == 'tensor_affine':
        scale = (xmax - xmin) / scale_denom 
        zero_point = (-q - (xmin / scale).int()).int()
        zero_point = torch.clamp(zero_point, -q, q - 1)
        
        return (torch.clamp(torch.round(x / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
    
    elif qscheme == 'tensor_mse':
        return quantize_tensor_mse(x, bits)
        
    else:
        raise NotImplementedError(qscheme)
        
        
# def quantize_tensor_wrapper(qscheme, bits):
#     def _wrapper(x, xmin=None, xmax=None):    
#         q = 2 ** (bits-1)
#         scale_denom = 2*q - 1
        
#         if xmin is None: xmin = x.min()
#         if xmax is None: xmax = x.max()
        
#         if qscheme == torch.per_tensor_symmetric:
#             scale = 2 * torch.where(xmin.abs() > xmax, xmin.abs(), xmax) / scale_denom
#             zero_point = torch.zeros(scale.shape).int()
                
#             return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale
        
#         elif qscheme == torch.per_tensor_affine:
#             scale = (xmax - xmin) / scale_denom 
#             zero_point = (-q - (xmin / scale).int()).int()
#             zero_point = torch.clamp(zero_point, -q, q - 1)
            
#             return (torch.clamp(torch.round(x / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
    
#     return _wrapper