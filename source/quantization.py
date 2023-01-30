import torch
import math

from source.utils import unfold


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


def get_tensor_stats(tensor, qscheme, mode = 0):  
    """Returns max and min along last dimension for mode-`mode` unfolding
    of `tensor` with modes starting at `0`, if qscheme is per channel.
    If qscheme is per tensor,  returns max and min computed across all elements.
    
    Parameters
    ----------
    tensor : ndarray
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    mode : int, default is 0
          Indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``.
    Returns
    -------
    tuple
        max, min of unfolded_tensor along last dimension.
    """
    if qscheme in ['channel_affine', 'channel_symmetric']:
        unfolded_tensor = unfold(tensor, mode = mode)

        tmax = unfolded_tensor.max(dim = -1)[0]
        tmin = unfolded_tensor.min(dim = -1)[0]
        #tmean = unfolded_tensor.mean(dim = -1)
        
    elif qscheme in ['tensor_affine', 'tensor_symmetric', 'tensor_log']:
        tmax = tensor.max()
        tmin = tensor.min()
        
    else:
        raise TypeError("Can't collect statistics. Unknown quantization scheme: {}".format(qscheme))
        return
   
    
    return tmax, tmin


def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1

    # compute min and max of input 
    min_val, max_val = input.min(), input.max()

    # transform input to [-1, 1] range - ?? not [0,1]? ?
    input_rescale = (input - min_val) / (max_val - min_val)

    # and quantize this rescaled input,
    n = math.pow(2.0, bits) - 1
    input_int = torch.floor(input_rescale * n + 0.5)

    # then compute quantized input
    quantized_input =  input_int * (max_val - min_val) / n + min_val
    return quantized_input 


def quantize_tensor_mse(x, bits, mse=False):
    def _quantize(x_float, maxval, mantissa_bits):

        exponent_bits = bits - 1 - mantissa_bits
        mantissa_bits = torch.tensor(mantissa_bits).to(x_float.device)
        exponent_bits = torch.tensor(exponent_bits).to(x_float.device)
        
        bias = 2 ** exponent_bits - torch.log2(maxval) + torch.log2(2 - 2 ** (-mantissa_bits)) - 1
        x_clipped = torch.min(torch.max(x_float, -maxval), maxval)

        log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()
        log_scales = torch.clamp(log_scales, 1.)

        scales = 2. ** (log_scales - mantissa_bits - bias)

        return torch.round(x_clipped / scales) * scales

    if bits == 8:
        mantissa_bits = 3
    elif bits == 6:
        mantissa_bits = 2
    elif bits == 4:
        mantissa_bits = 1
    else:
        raise NotImplementedError(bits)
    
    if mse:
    
        mx = torch.max(torch.abs(x.min()), torch.abs(x.max()))
        lsp = [torch.linspace(0.1 * mx.item(), 1.2 * mx.item(), 111)]
        # 111 x 1
        linspaces = torch.stack(lsp).to(x.device).transpose(0, 1)

        mses = torch.zeros_like(linspaces)

        meandims = list(torch.arange(len(x.shape)))

        for i, maxval in enumerate(linspaces):
            xfp = _quantize(x, maxval, mantissa_bits)
            mse = ((x - xfp) ** 2).mean(dim=meandims)
            mses[i, :] = mse

        best_mse = mses.argmin(0)
        maxval = torch.tensor([linspaces[best_mse[i], i] for i in range(linspaces.shape[-1])]).to(x.device)
        
    else:
        maxval = torch.max(torch.abs(x.min()), torch.abs(x.max()))

    return _quantize(x, maxval, mantissa_bits)


# def quantize_tensor(x, bits, qscheme):
#     q = 2 ** (bits-1)
#     scale_denom = 2*q - 1
        
#     xmin = x.min()
#     xmax = x.max()
        
#     if qscheme == 'tensor_symmetric':
#         scale = 2 * torch.where(xmin.abs() > xmax, xmin.abs(), xmax) / scale_denom

#         return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale
    
#     elif qscheme == 'tensor_affine':
#         scale = (xmax - xmin) / scale_denom 
#         zero_point = (-q - (xmin / scale).int()).int()
#         zero_point = torch.clamp(zero_point, -q, q - 1)
        
#         return (torch.clamp(torch.round(x / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
    
#     elif qscheme == 'tensor_mse':
#         return quantize_tensor_mse(x, bits, mse=True)
    
#     elif qscheme == 'tensor_log':
#         return quantize_tensor_mse(x, bits, mse=False)
    
#     elif qscheme == 'tensor_minmaxlog':
#         signs = torch.sign(x)
#         x_log = torch.log(torch.abs(x) + 1e-20)
        
#         quantized_x_log = min_max_quantize(x_log, bits-1)
        
#         return torch.exp(quantized_x_log) * signs
        
#     else:
#         raise NotImplementedError(qscheme)
        
        
def quantize_tensor(tensor, bits, qscheme, dim=None):
    """
    Parameters
    ----------
    tensor : Tensor
        Float tensor to quantize.
    dim : int or None, default is None
        If dim is not None, along the dimension `dim` the values in the `tensor` are scaled and offset by a different value (effectively the scale and offset become vectors).
        If dim is None, all values in the `tensor` are scaled and offset by the same value.
    
    Returns
    -------
    scale
        Scale to apply in quantization formula.
    zero_point
        Offset in integer value that maps to float zero.
    """
    
    #scale_denom = qmax - qmin, where qmax = 2**(nbits-1) - 1, qmin = -2**(nbits - 1)
    q = 2 ** (bits-1)
    scale_denom = 2*q - 1
    
    tmax, tmin = get_tensor_stats(tensor, qscheme, mode=dim)
    
    if qscheme in ['channel_symmetric', 'tensor_symmetric']:
        scale = 2 * torch.where(tmin.abs() > tmax, tmin.abs(), tmax) / scale_denom
        zero_point = torch.zeros(scale.shape).int()
        
        return torch.clamp(torch.round(tensor / scale), -q, q-1).to(int) * scale
        
    elif qscheme in ['channel_affine', 'tensor_affine']:
        scale = (tmax - tmin) / scale_denom 
        zero_point = (-q - (tmin / scale).int()).int()
        zero_point = torch.clamp(zero_point, -q, q - 1)
        
        return (torch.clamp(torch.round(tensor / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
        
    elif qscheme == 'tensor_mse':
        return quantize_tensor_mse(tensor, bits, mse=True)
    
    elif qscheme == 'tensor_log':
        return quantize_tensor_mse(tensor, bits, mse=False)
    
    elif qscheme == 'tensor_minmaxlog':
        signs = torch.sign(tensor)
        tensor_log = torch.log(torch.abs(tensor) + 1e-20)
        
        quantized_tensor_log = min_max_quantize(tensor_log, bits-1)
        
        return torch.exp(quantized_tensor_log) * signs
        
    else:
        raise NotImplementedError(qscheme)
