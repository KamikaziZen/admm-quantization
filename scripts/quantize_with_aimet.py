from flopco import FlopCo

import torch
import torch.nn as nn

from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

from source.data import get_imagenet_test_loader, get_imagenet_train_val_loaders
from source.eval import accuracy

from tqdm import tqdm
import numpy as np
import os
import random
from functools import partial
from collections import OrderedDict

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = [args.model_name]
    run_name.append(f"m={args.method}")
    run_name.append(f"b={args.bits}")
    run_name.append(f"r={args.reduction_rate}")
    run_name.append(f"i={args.init}")
    run_name.append(f"s={args.seed}")
    run_name.append(f"{args.qscheme}")
    run_name.append(f"calibrated_{args.calibration_samples}")
    run_name.append(f"w={args.param_bw}")
    run_name.append(f"a={args.output_bw}")
    if args.adaround: run_name.append("ada")
    
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--model-name",
                        type=str,
                        required=True)
    parser.add_argument("--data-root",
                        type=str,
                        help="Root dir of ImageNet Dataset")
    parser.add_argument("--method", 
                        type=str, 
                        required=True,
                        help="[admm, parafac, parafac-epc]")
    parser.add_argument("--init",
                        type=str,
                        required=False,
                        default='random',
                        help="[random, parafac-epc]")
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        help="Number of quantization bits.")
    parser.add_argument("--output_bw",
                        required=True, 
                        type=int,
                        help="Activations bitwidth.")
    parser.add_argument("--param_bw",
                        required=True, 
                        type=int,
                        help="Weights bitwidth.")
    parser.add_argument("--calibration-samples",
                        required=True, 
                        type=int,
                        help="Number of samples for calibration.")
    parser.add_argument("--batch-size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--adaround_samples",
                        required=False, 
                        type=int,
                        default=2048,
                        help="Number of samples for adaround.")
    parser.add_argument("--adaround_iterations",
                        required=False, 
                        default=20000,
                        type=int)
    parser.add_argument("--seed",
                        required=False, 
                        type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_mseminmax_symmetric, tensor_minmax]")
    parser.add_argument("--aimet-qscheme",
                        required=False,
                        default='tf-enhanced',
                        type=str)
    parser.add_argument("--reduction-rate",
                        required=False,
                        type=float)
    parser.add_argument("--adaround",
                        action="store_true")
    parser.add_argument("--num-workers",
                        type=int,
                        required=False,
                        default=4)
    parser.add_argument("--fold",
                        action='store_true')
    
    args = parser.parse_args()
    return args


def pass_calibration_data(sim_model, use_cuda, dataloader):
    batch_size = dataloader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for input_data, target_data in dataloader:

            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)

            batch_cntr += 1
            print(batch_cntr * batch_size)
            if (batch_cntr * batch_size) >= samples:
                break


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Running on: {device}')
    
    args = parse_args()
    logging.info(f'Args: {args}')
    
    set_seed(args.seed)
    
    if args.with_wandb:
        import wandb
        run = wandb.init(config=args, 
                         name=run_name(args))
    else:
        run=None
        
    # calibrated model
    model_name = run_name(args).split('_w=')[0]
    model = torch.load('checkpoints_stas/'+model_name)
    model = model.to(device)
    model.eval()
    
    # datasets
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
        
    # preparing model for quantization   
    model = prepare_model(model)
    if args.fold:
        _ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
        
    # counting macs and bops(macs * weight bitwidth * input bitwidth)
    model_stats = FlopCo(model, img_size=(1, 3, 224, 224), device=device,
                         instances=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d])
    macs = 0
    for x in model_stats.macs.values():
        macs += x[0]
    bops_stats = OrderedDict()
    bops = 0
    for name, macs_list in model_stats.macs.items():
        bops_stats[name]  = macs_list[0] * args.output_bw * args.param_bw
        bops += bops_stats[name]
    logging.info(str(bops_stats))
    logging.info(f'Macs: {macs}, Bops: {bops}')
    if run:
        wandb.log({'macs': macs, 'bops': bops})
        
    # quantization
    dummy_input = torch.rand(1, 3, 224, 224) # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
    dummy_input = dummy_input.to(device)
    
    if args.aimet_qscheme == 'tf_enhanced':
        aimet_qscheme = QuantScheme.post_training_tf_enhanced
    else:
        aimet_qscheme = QuantScheme.post_training_tf
    
    if args.adaround:
        params = AdaroundParameters(data_loader=val_loader, 
                                    num_batches=args.adaround_samples//val_loader.batch_size, 
                                    default_num_iterations=args.adaround_iterations)
        os.makedirs('adaround/', exist_ok=True)
        model = Adaround.apply_adaround(model, dummy_input, params,
                                    path='adaround/', 
                                    filename_prefix=model_name, 
                                    default_param_bw=args.param_bw,
                                    default_quant_scheme=aimet_qscheme)
        
    sim = QuantizationSimModel(model=model,
                               quant_scheme=aimet_qscheme,
                               dummy_input=dummy_input,
                               default_output_bw=args.output_bw,
                               default_param_bw=args.param_bw)
    
    if args.adaround:
        sim.set_and_freeze_param_encodings(
            encoding_path=os.path.join('adaround/', f'{model_name}.encodings'))
    
    sim.compute_encodings(forward_pass_callback=partial(pass_calibration_data, 
                                                        dataloader=train_loader),
                          forward_pass_callback_args=True)
    
    acc = accuracy(sim.model, test_loader, device=device)
    logging.info(f'Quantized model accuracy: {acc}')
    if run: 
        wandb.log({'acc_quantized': acc})
        run.finish()
        
    if run: run.finish()
    
    
if __name__ == '__main__':
    main()
