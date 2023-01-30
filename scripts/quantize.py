from flopco import FlopCo

import torch

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

from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = []
    run_name.append(f"m={args.method}")
    run_name.append(f"b={args.bits}")
    run_name.append(f"e={args.eps}")
    run_name.append(f"d={args.decomp}")
    run_name.append(f"s={args.seed}")
    run_name.append(f"{args.qscheme}")
    run_name.append(f"calibrated_{args.calibration_samples}")
    if args.no_layer1: run_name.append("no_layer1")
    run_name.append(f"w={args.param_bw}")
    run_name.append(f"a={args.output_bw}")
    if args.adaround: run_name.append("ada")
    
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--with_wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--method", 
                        type=str, 
                        required=True,
                        help="[admm, parafac]")
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
    parser.add_argument("--calibration_samples",
                        required=True, 
                        type=int,
                        help="Number of samples for calibration.")
    parser.add_argument("--batch_size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--adaround_samples",
                        required=True, 
                        type=int,
                        help="Number of samples for adaround.")
    parser.add_argument("--adaround_iterations",
                        required=False, 
                        default=20000,
                        type=int)
    parser.add_argument("--seed",
                        required=True, 
                        type=int,
                        help="Random seed.")
    parser.add_argument("--qscheme",
                        required=True,
                        type=str,
                        help="[tensor_symmetric, tensor_affine, tensor_mse, tensor_minmaxlog]")
    parser.add_argument("--eps",
                        required=True,
                        type=float)
    parser.add_argument("--decomp",
                        required=True,
                        type=str)
    parser.add_argument("--no_layer1",
                        action="store_true",
                        help='If True, layer1.0.conv1 is not factorized.')
    parser.add_argument("--adaround",
                        action="store_true")
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
    model = torch.load('checkpoints/'+model_name)
    model = model.to(device)
    model.eval()
    
    # datasets
    train_loader, val_loader = get_imagenet_train_val_loaders(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/',
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       val_perc=0.04,
                                       shuffle=True,
                                       random_seed=args.seed)
    test_loader = get_imagenet_test_loader(data_root='/gpfs/gpfs0/k.sobolev/ILSVRC-12/', 
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True,
                                       shuffle=False)
    
    # factorized model macs
    model_stats = FlopCo(model.to(device), img_size=(1, 3, 224, 224), device=device)
    redc_macs = 0
    for x in model_stats.macs.values():
        redc_macs += x[0]
    redc_macs_pc = redc_macs / 1814073344
    logging.info(f'Reduced Macs: {redc_macs_pc}')    
    if run: wandb.log({'macs_reduced': redc_macs_pc})
        
    # % BOPs = output_bw * MACs * param_bw
    bops = args.param_bw / 32 * args.output_bw / 32 * redc_macs_pc * 100
    logging.info(f'%BOPS: {bops}')
    if run: wandb.log({'bops': bops})
        
    # quantization    
    model = prepare_model(model)
    _ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
    dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
    dummy_input = dummy_input.to(device)
    if args.adaround:
        params = AdaroundParameters(data_loader=val_loader, 
                                    num_batches=args.adaround_samples//val_loader.batch_size, 
                                    default_num_iterations=args.adaround_iterations)
        os.makedirs('adaround/', exist_ok=True)
        model = Adaround.apply_adaround(model, dummy_input, params,
                                    path='adaround/', 
                                    filename_prefix=model_name, 
                                    default_param_bw=args.param_bw,
                                    default_quant_scheme=QuantScheme.post_training_tf_enhanced)
        
    sim = QuantizationSimModel(model=model,
                               quant_scheme=QuantScheme.post_training_tf_enhanced,
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