import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18
import numpy as np
import random

from flopco import FlopCo
from musco.pytorch.compressor.config_gen import generate_model_compr_kwargs
from musco.pytorch import Compressor
from musco.pytorch.compressor.rank_estimation.estimator import estimate_rank_for_compression_rate

import copy
import gc
from collections import defaultdict
import argparse
from tqdm.notebook import tqdm

from source.data import get_imagenet_train_val_loaders, get_imagenet_test_loader
from source.eval import accuracy, estimate_macs
from source.utils import get_layer_by_name, bncalibrate_layer

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_name(args):
    run_name = ['resnet18']
    run_name.append(f"l={args.lname}")
    run_name.append(f"eps={args.eps:.3f}")
    run_name.append(f"{args.decomposition}")
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--with_wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--lname", 
                        type=str, 
                        required=True,
                        help="Name of the layer.")
    parser.add_argument("--data_root", 
                        type=str, 
                        required=True,
                        help="Path to stored dataset.")
    parser.add_argument("--decomposition", 
                        type=str, 
                        required=True,
                        help="One of [cp3, tucker, svd, cp3-epc].")
    parser.add_argument("--eps",
                        required=True, 
                        type=float,
                        help="Allowed drop in accuracy.")
    parser.add_argument("--batch_size",
                        required=False, 
                        type=int,
                        default=500,
                        help="Batch size for evaluation and calibration.")
    parser.add_argument("--num_workers",
                        required=False, 
                        type=int,
                        default=4,
                        help="Number of workers for dataloaders.")
    parser.add_argument("--start_reduction_rate",
                        required=False, 
                        type=int,
                        help="Reduction rate from which to start binary search.")
    parser.add_argument("--min_rank",
                        required=False, 
                        type=int,
                        help="Minimum rank to start binary search.")
    parser.add_argument("--max_rank",
                        required=False, 
                        type=int,
                        help="Maximum rank to start binary search.")
    parser.add_argument("--seed",
                        required=False, 
                        type=int,
                        default=42,
                        help="Random seed.")

    args = parser.parse_args()
    # Sanity Check
    if args.decomposition not in ['cp3', 'tucker2', 'svd', 'cp3-epc']:
        raise ValueError('Wrong decomposition name. Correct options: (cp3, tucker2, svd, cp3-epc)')
        
    return args


def find_best_rank_for_layer(model, lname, decomposition, train_loader, val_loader, 
                             eval_func, bn_cal_func, bn_cal_n_iters, score_eps, 
                             max_rank, min_rank=3, grid_step=1, nx=1, device='cuda'):
    '''
    Find minimal decomposition rank for given acceptable target metric drop (uses binary search)
    Parameters:
    model           -   Initial model
    lname           -   Name of layer to find decomposition rank, String
    decomposition   -   Decomposition algorithm name, Options: (cp3, tucker2, svd), String
    score_eps       -   Acceptable target metric drop, float
    train_loader    -   Training dataset dataloader, Pytorch Dataloader
    val_loader      -   Validation dataset dataloader, Pytorch Dataloader
    eval_func       -   Function for model evaluation (returns target metric score,
                        args: temp_model, val_loader, device), Python function
    bn_cal_func     -   Function for batchnorm statistics calibration
                        (args: emp_model, train_loader, lname, bn_cal_n_iters, device), Python function
    bn_cal_n_iters  -   Number of batchnorm callibration iterations, int
    max_rank        -   Upper bound of rank search, int
    min_rank        -   Lower bound of rank search, int
    grid_step       -   Rank search grid step (search for ranks multiple of grid_step)
    nx              -   Minimal compression ratio for layer FLOPs, float
    device          -   Device to store the model
    
    Output:
    best_rank       -   Best rank for compression of given layer, int or None
                        (if layer can not be compressed with given settings)
    '''
    
    if decomposition not in ['cp3', 'tucker2', 'svd', 'cp3-epc']:
        raise ValueError('Wrong decomposition name. Correct options: (cp3, tucker2, svd, cp3-epc)')
    
    curr_rank = max_rank // grid_step if max_rank // grid_step != 0 else 1
    curr_max = max_rank // grid_step if max_rank // grid_step != 0 else 1
    curr_min = min_rank // grid_step if min_rank // grid_step != 0 else 1
    best_rank = None

    n = int(np.log2(curr_max)) + 1
    score_init = eval_func(model.to(device), val_loader, device=device)
    
    init_layer = get_layer_by_name(model, lname)
    ch_ratio = init_layer.in_channels / init_layer.out_channels
    
    if curr_max < curr_min:
        logging.error("Layer can not be compressed with given grid step")
    for i in range(n):
        logging.info("Search iter {}: ranks (min, curr, max): ({}, {}, {})".format(i, curr_min, curr_rank, 
                                                                            curr_max))

        logging.info("-------------------------\n Compression step")
        
        manual_rank = (int(curr_rank * ch_ratio), curr_rank) if decomposition=='tucker2' else curr_rank
        
        model_compr_kwargs = {lname: {'decomposition': decomposition,
                                      'rank_selection': 'manual',
                                      'manual_rank': [manual_rank],
                                      'curr_compr_iter': 0}
                              }
        model_stats = FlopCo(model.to(device), img_size=(1, 3, 224, 224), device=device)

        compressor = Compressor(copy.deepcopy(model.cpu()),
                                model_stats,
                                ft_every=3,
                                nglobal_compress_iters=1,
                                model_compr_kwargs = model_compr_kwargs,
                               )
        compressor.compression_step()

        logging.info("-------------------------\n Calibration step")
        # calibrate batch norm statistics

        compressor.compressed_model.to(device)
        bn_cal_func(compressor.compressed_model, train_loader, layer_name=lname,
                    n_batches=bn_cal_n_iters, device=device)

        logging.info("-------------------------\n Test step")

        # eval model
        score = eval_func(compressor.compressed_model, val_loader, device=device)
        logging.info('Current score: {}'.format(score))

        # clear memory
        del compressor
        gc.collect()
        torch.cuda.empty_cache()

        if score + score_eps < score_init:

            if i == 0:
                logging.error("Bad layer to compress")
                if nx > 1:
                    best_rank = curr_rank
                break
            else:
                curr_min = curr_rank
                curr_rank = curr_rank + (curr_max - curr_rank) // 2
        else:
            best_rank = curr_rank

            curr_max = curr_rank
            curr_rank = curr_rank - (curr_rank - curr_min) // 2

    if best_rank is not None:
        return best_rank * grid_step
    else:
        return best_rank

    
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Running on: {device}')
    
    
    args = parse_args()
    set_seed(args.seed)
    logging.info(f'Args: {args}')
    
    if args.with_wandb:
        import wandb
        run = wandb.init(config=args, 
                         name=run_name(args))
    else:
        run=None
    
    model = resnet18(pretrained=True).to(device)
    model.eval()
    
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
    
    model_stats = FlopCo(model, img_size=(1, 3, 224, 224), device=device)
    all_lnames = list(model_stats.flops.keys())
    # create list of layer names to be compressed
    lnames_to_compress = [k for k in all_lnames if model_stats.ltypes[k]['type'] == nn.Conv2d \
                          and k != 'conv1' and 'downsample' not in k]
    max_ranks = {}

    for lname in lnames_to_compress:
        layer_shape = get_layer_by_name(model, lname).weight.shape
        if args.max_rank:
            max_ranks[lname] = args.max_rank
        else:
            max_ranks[lname] = estimate_rank_for_compression_rate(layer_shape, 
                                                                  rate=args.start_reduction_rate,
                                                                  tensor_format='cp3')

    saved_ranks = {k: None for k in all_lnames}
    min_ranks = {k: args.min_rank or 10 for k in max_ranks.keys()}
    curr_ranks = copy.deepcopy(max_ranks)
    
    logging.info(f'Layer: {args.lname}')
    logging.info(f'Decomposition: {args.decomposition}')
    logging.info(f'Eps: {args.eps}')
    
    best_rank = find_best_rank_for_layer(model, 
                         lname=args.lname, 
                         decomposition=args.decomposition, 
                         train_loader=train_loader, 
                         val_loader=val_loader, 
                         eval_func=accuracy,
                         bn_cal_func=bncalibrate_layer, 
                         bn_cal_n_iters=1, 
                         score_eps=args.eps,
                         max_rank=max_ranks[args.lname], 
                         min_rank=min_ranks[args.lname],
                         grid_step=1, 
                         device=device)
    logging.info('Best rank:', best_rank)
    
    orig_macs, redc_macs = estimate_macs(model, args.lname, best_rank, device='cpu')
    logging.info(f'Original macs: {orig_macs}')
    logging.info(f'Reduced macs: {redc_macs}')
    logging.info(f'Redc/Orig: {redc_macs/orig_macs}')
    
    if run:
        wandb.log({'best_rank': best_rank, 'orig_macs': orig_macs,
                  'redc_macs': redc_macs, 'redc/orig': redc_macs / orig_macs})
        run.finish()
        

if __name__ == '__main__':
    main()
