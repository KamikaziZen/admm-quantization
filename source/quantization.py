import torch
from torch import nn
from torch.quantization import DeQuantStub

import math
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional, Tuple, Dict, Mapping

from source.utils import unfold


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


def min_max_quantize(input, bits, min_val=None, max_val=None):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1

    # compute min and max of input 
    if min_val is None or max_val is None:
        min_val, max_val = input.min(), input.max()

    # transform input to [0,1]
    input_rescale = (input - min_val) / (max_val - min_val)

    # and quantize this rescaled input,
    n = math.pow(2.0, bits) - 1
    input_int = torch.floor(input_rescale * n + 0.5)

    # then compute quantized input
    quantized_input =  input_int * (max_val - min_val) / n + min_val
    return quantized_input 
        
        
def quantize_tensor(tensor, bits, qscheme, dim=None, **kwargs):
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
    
    if qscheme in ['channel_symmetric', 'tensor_symmetric']:
        tmax, tmin = get_tensor_stats(tensor, qscheme, mode=dim)
        scale = 2 * torch.where(tmin.abs() > tmax, tmin.abs(), tmax) / scale_denom
        
        return torch.clamp(torch.round(tensor / scale), -q, q-1).to(int) * scale
        
    elif qscheme in ['channel_affine', 'tensor_affine']:
        if kwargs.get('tmin') is None or kwargs.get('tmax') is None:
            tmax, tmin = get_tensor_stats(tensor, qscheme, mode=dim)
        else:
            tmax, tmin = kwargs['tmax'], kwargs['tmin']
        scale = (tmax - tmin) / scale_denom 
        zero_point = (-q - (tmin / scale).int()).int()
        zero_point = torch.clamp(zero_point, -q, q - 1)
        
        return (torch.clamp(torch.round(tensor / scale) + zero_point, -q, q-1).to(int) - zero_point) * scale
        
    elif qscheme == 'tensor_mseminmax_symmetric':
        return quantize_tensor_mse(tensor, bits, **kwargs)
    
    elif qscheme == 'tensor_minmax':
        return min_max_quantize(tensor, bits)
        
    else:
        raise NotImplementedError(qscheme)

        
def quantize_tensor_mse(x, bits, num_attempts=200):
    
    q = 2 ** (bits-1)
    scale_denom = 2*q - 1
    
    def _quantize(x, tmax):
        scale = 2 * tmax / scale_denom 
#         return torch.clamp(torch.round(x / scale), -q, q-1).to(int) * scale
        return torch.clamp(torch.round(x / scale), -q, q-1) * scale
        
    
    mx = torch.max(torch.abs(x.min()), torch.abs(x.max()))
    lsp = [torch.linspace(0.2 * mx.item(), 1.2 * mx.item(), num_attempts)]
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


def fixed_quantize(input, bits, zero_point, scope):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1

    # transform input to [0,1]
    input_rescale = (input - zero_point) / scope

    # and quantize this rescaled input,
    n = math.pow(2.0, bits) - 1
    input_int = torch.floor(input_rescale * n + 0.5)

    # then compute quantized input
    quantized_input =  input_int * scope / n + zero_point
    return quantized_input 

# def fixed_quantize(input, bits, zero_point, scope):
#     scale = scope / (math.pow(2.0, bits) - 1)
#     return scale * (torch.round(input / scale - torch.round(torch.tensor(zero_point / scale))) + torch.round(torch.tensor(zero_point / scale)))


class FixedQuant(nn.Module):
    def __init__(self, name, bits, **kwargs):
        super().__init__()
        self.name = name
        self.bits = bits
        self.quant_func = fixed_quantize
        self.zero_point = kwargs['zero_point']
        self.scope = kwargs['scope']

    def forward(self, x_orig):
        if self.zero_point is not None and self.scope is not None:
            output = self.quant_func(x_orig, self.bits, self.zero_point, self.scope)
        else:
            output = x_orig
        return output

    def __repr__(self):
        return '{}(bits={} zero_point={} scope={})'.format(self.__class__.__name__, 
                                                           self.bits, self.zero_point, self.scope)
    

class MinMaxQuant(nn.Module):
    def __init__(self, name, bits, counter, **kwargs):
        super().__init__()
        self.name = name
        self.bits = bits
        self.quant_func = min_max_quantize
        self._counter = counter
        self.min_val = None
        self.max_val = None

    @property
    def counter(self):
        return self._counter

    def forward(self, x_orig):
        if self._counter > 0:
            self._counter -= len(x_orig)
            
            if x_orig.numel() == 0:
                return x_orig
            
            x = x_orig.detach()  # avoid keeping autograd tape
            min_val_cur, max_val_cur = torch.aminmax(x)
            if self.min_val is not None:
                self.min_val.copy_(torch.min(min_val_cur, self.min_val))
            else:
                self.min_val = torch.tensor(min_val_cur.item(), dtype=torch.float)
            if self.max_val is not None:
                self.max_val.copy_(torch.max(max_val_cur, self.max_val))
            else:
                self.max_val = torch.tensor(max_val_cur.item(), dtype=torch.float)
            return x_orig
        else:
            output = self.quant_func(x_orig, self.bits, self.min_val, self.max_val)
            return output

    def __repr__(self):
        return '{}(bits={} min_val={} max_val={})'.format(self.__class__.__name__, 
                                                          self.bits, self.min_val, self.max_val)
    
    
class MovingAvgMinMaxQuant(nn.Module):
    def __init__(self, name, bits, counter, **kwargs):
        super().__init__()
        self.name = name
        self.bits = bits
        self.quant_func = min_max_quantize
        self._counter = counter
        self.min_val = None
        self.max_val = None
        self.averaging_constant = 0.01

    @property
    def counter(self):
        return self._counter

    def forward(self, x_orig):
        if self._counter > 0:
            self._counter -= len(x_orig)
            
            if x_orig.numel() == 0:
                return x_orig
            
            x = x_orig.detach()  # avoid keeping autograd tape
            min_val_cur, max_val_cur = torch.aminmax(x)
            if self.min_val is not None:
                min_val = self.min_val + self.averaging_constant * (min_val_cur - self.min_val)
                self.min_val.copy_(min_val)
            else:
                self.min_val = torch.tensor(min_val_cur.item(), dtype=torch.float)
            if self.max_val is not None:
                max_val = self.max_val + self.averaging_constant * (max_val_cur - self.max_val)
                self.max_val.copy_(max_val)
            else:
                self.max_val = torch.tensor(max_val_cur.item(), dtype=torch.float)
            return x_orig
        else:
            output = self.quant_func(x_orig, self.bits, self.min_val, self.max_val)
            return output

    def __repr__(self):
        return '{}(bits={} min_val={} max_val={})'.format(self.__class__.__name__, 
                                                          self.bits, self.min_val, self.max_val)


def get_quant_layer(ltype, name, **kwargs):
    if ltype == 'minmax':
        quant_layer = MinMaxQuant('{}_quant'.format(name), **kwargs)
    elif ltype == 'avgminmax':
        quant_layer = MovingAvgMinMaxQuant('{}_quant'.format(name), **kwargs)
    elif ltype == 'histogram':
        quant_layer = HistogramQuant('{}_quant'.format(name), **kwargs)
    elif ltype == 'fixed':
        quant_layer = FixedQuant('{}_quant'.format(name), **kwargs)
    else:
        raise NotImplementedError(ltype)
            
    return quant_layer


def add_quant(name, module, ltype, **kwargs):
    mlist = OrderedDict()
    if isinstance(module, nn.Sequential):
        for km, m in module._modules.items():
            mlist[km] = m
    else:
        mlist[name] = module
    mlist['activation_post_process'] = get_quant_layer(ltype, name, **kwargs)
    
    return nn.Sequential(mlist)
    
    
def duplicate_resnet_with_quant(model, ltype, **kwargs):
    
    for km, m in model._modules.items():
        if isinstance(m, nn.Sequential):
            for i in range(len(m)):
                for kl, l in m[i]._modules.items():
                    if kl == 'downsample':
                        l[0] = add_quant('0', l[0], ltype, **kwargs)
                        l[1] = add_quant('1', l[1], ltype, **kwargs)
                    elif kl == 'ff':
                        l.__setattr__('activation_post_process', get_quant_layer(ltype, 'ff', **kwargs))
                    elif not isinstance(l, nn.ReLU):
                        m[i].__setattr__(kl, add_quant(kl, l, ltype, **kwargs))
        elif not isinstance(m, (DeQuantStub, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            model.__setattr__(km, add_quant(km, m, ltype, **kwargs))
            
    return model


def duplicate_resnet_factorized_with_quant(model, ltype, **kwargs):
    
    for km, m in model._modules.items():
        if km == 'conv1' and isinstance(m, nn.Sequential):
            m.__setattr__('conv1', add_quant('conv1', m.conv1, ltype, **kwargs))
            m.__setattr__('conv2', add_quant('conv2', m.conv2, ltype, **kwargs))
            m.__setattr__('conv3', add_quant('conv3', m.conv3, ltype, **kwargs))
        elif isinstance(m, nn.Sequential):
            for i in range(len(m)):
                for kl, l in m[i]._modules.items():
                    if kl == 'downsample':
                        l[0] = add_quant('0', l[0], ltype, **kwargs)
                        l[1] = add_quant('1', l[1], ltype, **kwargs)
                    elif kl == 'ff':
                        l.__setattr__('activation_post_process', get_quant_layer(ltype, 'ff', **kwargs))
                    elif isinstance(l, nn.Sequential):
                        l.__setattr__('conv1', add_quant('conv1', l.conv1, ltype, **kwargs))
                        l.__setattr__('conv2', add_quant('conv2', l.conv2, ltype, **kwargs))
                        if hasattr(l, 'conv3'):
                            l.__setattr__('conv3', add_quant('conv3', l.conv3, ltype, **kwargs))
                    elif not isinstance(l, nn.ReLU):
                        m[i].__setattr__(kl, add_quant(kl, l, ltype, **kwargs))
        elif not isinstance(m, (DeQuantStub, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            model.__setattr__(km, add_quant(km, m, ltype, **kwargs))
            
    return model


class HistogramQuant(nn.Module):
    def __init__(self, name, bits, counter, **kwargs):
        super().__init__()
        self.name = name
        self.bits = bits
        self.quant_func = min_max_quantize
        self._counter = counter
        self.min_val = None
        self.max_val = None
        self.bins = 2048
        self.upsample_rate = 128
        self.histogram = torch.zeros(self.bins)
        self.dst_nbins = 2 ** self.bits
        self.calculated = False

    @property
    def counter(self):
        return self._counter
    
    def _adjust_min_max(self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int):
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.
        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = int(
            torch.ceil(
                (combined_max - combined_min) / (self.bins * hist_bin_width)
            ).item()
        )
        e = downsample_rate * (self.bins * hist_bin_width) - (
            combined_max - combined_min
        )
        # Relax only the max, not the min, so that for one sided distributions, min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = int(
            torch.round((self.min_val - combined_min) / hist_bin_width).item()
        )
        return combined_min, combined_max, downsample_rate, start_idx
    
    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm
    
    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()
    
    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        new_hist: torch.Tensor,
        upsample_rate: int,
        downsample_rate: int,
        start_idx: int,
        Nbins: int,
    ) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist
        
    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max
        
    def _calculate_qparams(self):
        is_uninitialized = self.min_val is None and self.max_val is None
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0], device=self.min_val.device.type)
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()
        self.min_val.copy_(new_min)
        self.max_val.copy_(new_max)
        
        self.calculated = True
        
    def forward(self, x_orig):
        if self._counter > 0:
            self._counter -= len(x_orig)
            
            is_uninitialized = self.min_val is None and self.max_val is None
            
            if x_orig.numel() == 0:
                return x_orig
            
            x = x_orig.detach()  # avoid keeping autograd tape
            min_val_cur, max_val_cur = torch.aminmax(x)
            
            if is_uninitialized or self.min_val.item() == self.max_val.item():
                self.min_val = torch.tensor(min_val_cur.item(), dtype=torch.float)
                self.max_val = torch.tensor(max_val_cur.item(), dtype=torch.float)
            else:
                combined_min = torch.min(min_val_cur, self.min_val)
                combined_max = torch.max(max_val_cur, self.max_val)
                (
                    combined_min,
                    combined_max,
                    downsample_rate,
                    start_idx,
                ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
                assert (
                    combined_min.numel() == 1 and combined_max.numel() == 1
                ), "histogram min/max values must be scalar."
                
            
                combined_histogram = torch.histc(
                    x, self.bins, min=int(combined_min), max=int(combined_max)
                )
                if combined_min == self.min_val and combined_max == self.max_val:
                    self.histogram = self.histogram.to(combined_histogram)
                    combined_histogram += self.histogram
                else:
                    combined_histogram = self._combine_histograms(
                        combined_histogram,
                        self.histogram,
                        self.upsample_rate,
                        downsample_rate,
                        start_idx,
                        self.bins,
                    )

                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(combined_min.shape)
                self.min_val.copy_(combined_min)
                self.max_val.detach_().resize_(combined_max.shape)
                self.max_val.copy_(combined_max)
            
#             print(self.name, self.min_val, self.max_val)
            return x_orig
        else:
        
            if self.calculated is False:
                self._calculate_qparams()
                
            output = self.quant_func(x_orig, self.bits, self.min_val, self.max_val)
#             output = quantize_tensor(x_orig, self.bits, 'tensor_affine', 
#                                      tmin=self.min_val, tmax=self.max_val)
            return output

    def __repr__(self):
        return '{}(bits={} min_val={} max_val={} calculated={})'.format(self.__class__.__name__, 
            self.bits, self.min_val, self.max_val, self.calculated)