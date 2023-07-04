import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from skimage import io

import numpy as np
import os
import pickle
import random

# from source.utils import crop_sample, pad_sample, resize_sample, normalize_volume


def get_imagenet_train_val_loaders(batch_size=128,
                                   val_perc=0.1,
                                   data_root=None,
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=True,
                                   random_seed=None):
    '''  Returns iterators through train/val CIFAR10 datasets.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param batch_size: int
        How many samples per batch to load.
    :param val_perc: float
        Percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the train/validation indices
    :param random_seed: int
        Fix seed for reproducibility.
    :return:
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

    traindir = os.path.join(data_root, 'train')

    train_dataset = datasets.ImageFolder(traindir, transform_train)
    val_dataset = datasets.ImageFolder(traindir, transform_test)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_perc * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              drop_last=True, pin_memory=pin_memory, )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                            drop_last=True, pin_memory=pin_memory, )

    return train_loader, val_loader


def get_imagenet_test_loader(batch_size=128, data_root=None, num_workers=1, pin_memory=True, shuffle=False):
    ''' Returns iterator through CIFAR10 test dataset

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param batch_size: int
        How many samples per batch to load.
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the dataset after every epoch.
    :return:
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

    test_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             drop_last=True, pin_memory=pin_memory)
    return test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
            

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

    
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, 
    

def get_cifar100_train_loader(data_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_cifar100_test_loader(data_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


# class BrainSegmentationDataset(Dataset):
#     """Brain MRI dataset for FLAIR abnormality segmentation"""

#     in_channels = 3
#     out_channels = 1

#     def __init__(
#         self,
#         images_dir,
#         transform=None,
#         image_size=256,
#         subset="train",
#         random_sampling=True,
#         validation_cases=10,
#         seed=42,
#     ):
#         assert subset in ["all", "train", "validation"]

#         # read images
#         volumes = {}
#         masks = {}
#         print("reading {} images...".format(subset))
#         for (dirpath, dirnames, filenames) in os.walk(images_dir):
#             image_slices = []
#             mask_slices = []
#             for filename in sorted(
#                 filter(lambda f: ".tif" in f, filenames),
#                 key=lambda x: int(x.split(".")[-2].split("_")[4]),
#             ):
#                 filepath = os.path.join(dirpath, filename)
#                 if "mask" in filename:
#                     mask_slices.append(io.imread(filepath, as_gray=True))
#                 else:
#                     image_slices.append(io.imread(filepath))
#             if len(image_slices) > 0:
#                 patient_id = dirpath.split("/")[-1]
#                 volumes[patient_id] = np.array(image_slices[1:-1])
#                 masks[patient_id] = np.array(mask_slices[1:-1])

#         self.patients = sorted(volumes)

#         # select cases to subset
#         if not subset == "all":
#             random.seed(seed)
#             validation_patients = random.sample(self.patients, k=validation_cases)
#             if subset == "validation":
#                 self.patients = validation_patients
#             else:
#                 self.patients = sorted(
#                     list(set(self.patients).difference(validation_patients))
#                 )

#         print("preprocessing {} volumes...".format(subset))
#         # create list of tuples (volume, mask)
#         self.volumes = [(volumes[k], masks[k]) for k in self.patients]

#         print("cropping {} volumes...".format(subset))
#         # crop to smallest enclosing volume
#         self.volumes = [crop_sample(v) for v in self.volumes]

#         print("padding {} volumes...".format(subset))
#         # pad to square
#         self.volumes = [pad_sample(v) for v in self.volumes]

#         print("resizing {} volumes...".format(subset))
#         # resize
#         self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

#         print("normalizing {} volumes...".format(subset))
#         # normalize channel-wise
#         self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

#         # probabilities for sampling slices based on masks
#         self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
#         self.slice_weights = [
#             (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
#         ]

#         # add channel dimension to masks
#         self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

#         print("done creating {} dataset".format(subset))

#         # create global index for patient and slice (idx -> (p_idx, s_idx))
#         num_slices = [v.shape[0] for v, m in self.volumes]
#         self.patient_slice_index = list(
#             zip(
#                 sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
#                 sum([list(range(x)) for x in num_slices], []),
#             )
#         )

#         self.random_sampling = random_sampling

#         self.transform = transform

#     def __len__(self):
#         return len(self.patient_slice_index)

#     def __getitem__(self, idx):
#         patient = self.patient_slice_index[idx][0]
#         slice_n = self.patient_slice_index[idx][1]

#         if self.random_sampling:
#             patient = np.random.randint(len(self.volumes))
#             slice_n = np.random.choice(
#                 range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
#             )

#         v, m = self.volumes[patient]
#         image = v[slice_n]
#         mask = m[slice_n]

#         if self.transform is not None:
#             image, mask = self.transform((image, mask))

#         # fix dimensions (C, H, W)
#         image = image.transpose(2, 0, 1)
#         mask = mask.transpose(2, 0, 1)

#         image_tensor = torch.from_numpy(image.astype(np.float32))
#         mask_tensor = torch.from_numpy(mask.astype(np.float32))

#         # return tensors
#         return image_tensor, mask_tensor
    
    
# def get_mri_test_loader(data_root, batch_size, drop_last=True, num_workers=1):
#     dataset = BrainSegmentationDataset(
#         images_dir=data_root,
#         subset="validation",
#         image_size=256,
#         random_sampling=False,
#     )
#     loader = DataLoader(
#         dataset, batch_size=batch_size, 
#         drop_last=drop_last, num_workers=num_workers
#     )
#     return loader