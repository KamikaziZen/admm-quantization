import numpy as np
import torch
import torch.nn as nn

import enum

from tqdm import tqdm
from typing import Optional, Type, TypeVar
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, Type

# from medpy.filter.binary import largest_connected_component
# from skimage.exposure import rescale_intensity
# from skimage.transform import resize


T = TypeVar("T", bound=enum.Enum)
V = TypeVar("V")


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


class StrEnumMeta(enum.EnumMeta):
    auto = enum.auto

    def from_str(self: Type[T], member: str) -> T:  # type: ignore[misc]
        try:
            return self[member]
        except KeyError:
            # TODO: use `add_suggestion` from torchvision.prototype.utils._internal to improve the error message as
            #  soon as it is migrated.
            raise ValueError(f"Unknown value '{member}' for {self.__name__}.") from None        

            
class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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

    # Set BN and LN to train regime
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
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


# def dsc(y_pred, y_true, lcc=True):
#     if lcc and np.any(y_pred):
#         y_pred = np.round(y_pred).astype(int)
#         y_true = np.round(y_true).astype(int)
#         y_pred = largest_connected_component(y_pred)
#     return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


# def crop_sample(x):
#     volume, mask = x
#     volume[volume < np.max(volume) * 0.1] = 0
#     z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
#     z_nonzero = np.nonzero(z_projection)
#     z_min = np.min(z_nonzero)
#     z_max = np.max(z_nonzero) + 1
#     y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
#     y_nonzero = np.nonzero(y_projection)
#     y_min = np.min(y_nonzero)
#     y_max = np.max(y_nonzero) + 1
#     x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
#     x_nonzero = np.nonzero(x_projection)
#     x_min = np.min(x_nonzero)
#     x_max = np.max(x_nonzero) + 1
#     return (
#         volume[z_min:z_max, y_min:y_max, x_min:x_max],
#         mask[z_min:z_max, y_min:y_max, x_min:x_max],
#     )


# def pad_sample(x):
#     volume, mask = x
#     a = volume.shape[1]
#     b = volume.shape[2]
#     if a == b:
#         return volume, mask
#     diff = (max(a, b) - min(a, b)) / 2.0
#     if a > b:
#         padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
#     else:
#         padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
#     mask = np.pad(mask, padding, mode="constant", constant_values=0)
#     padding = padding + ((0, 0),)
#     volume = np.pad(volume, padding, mode="constant", constant_values=0)
#     return volume, mask


# def resize_sample(x, size=256):
#     volume, mask = x
#     v_shape = volume.shape
#     out_shape = (v_shape[0], size, size)
#     mask = resize(
#         mask,
#         output_shape=out_shape,
#         order=0,
#         mode="constant",
#         cval=0,
#         anti_aliasing=False,
#     )
#     out_shape = out_shape + (v_shape[3],)
#     volume = resize(
#         volume,
#         output_shape=out_shape,
#         order=2,
#         mode="constant",
#         cval=0,
#         anti_aliasing=False,
#     )
#     return volume, mask


# def normalize_volume(volume):
#     p10 = np.percentile(volume, 10)
#     p99 = np.percentile(volume, 99)
#     volume = rescale_intensity(volume, in_range=(p10, p99))
#     m = np.mean(volume, axis=(0, 1, 2))
#     s = np.std(volume, axis=(0, 1, 2))
#     volume = (volume - m) / s
#     return volume


# def log_images(x, y_true, y_pred, channel=1):
#     images = []
#     x_np = x[:, channel].cpu().numpy()
#     y_true_np = y_true[:, 0].cpu().numpy()
#     y_pred_np = y_pred[:, 0].cpu().numpy()
#     for i in range(x_np.shape[0]):
#         image = gray2rgb(np.squeeze(x_np[i]))
#         image = outline(image, y_pred_np[i], color=[255, 0, 0])
#         image = outline(image, y_true_np[i], color=[0, 255, 0])
#         images.append(image)
#     return images


# def gray2rgb(image):
#     w, h = image.shape
#     image += np.abs(np.min(image))
#     image_max = np.abs(np.max(image))
#     if image_max > 0:
#         image /= image_max
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
#     return ret


# def outline(image, mask, color):
#     mask = np.round(mask)
#     yy, xx = np.nonzero(mask)
#     for y, x in zip(yy, xx):
#         if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
#             image[max(0, y) : y + 1, max(0, x) : x + 1] = color
#     return image