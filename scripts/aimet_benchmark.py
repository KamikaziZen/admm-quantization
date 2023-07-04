from flopco import FlopCo

import torch
from torchvision.models import resnet18, resnet34, resnet50

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
import time
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
    run_name = [args.model_name]
    run_name.append(f"s={args.seed}")
    run_name.append(f"w={args.param_bw}")
    run_name.append(f"a={args.output_bw}")
    if args.adaround: run_name.append("ada")
    
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-name",
                        type=str,
                        help="[resnet18, resnet34, resnet50]")
    parser.add_argument("--data-root",
                        type=str,
                        help="Root dir of ImageNet Dataset")
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--output-bw",
                        required=True, 
                        type=int,
                        help="Activations bitwidth.")
    parser.add_argument("--param-bw",
                        required=True, 
                        type=int,
                        help="Weights bitwidth.")
    parser.add_argument("--qscheme",
                        required=False,
                        default='tf_enhanced',
                        type=str,
                        help='Use tf for usual min-max and tf_enhanced for sqnr-optimized min-max.')
    parser.add_argument("--batch-size",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument("--adaround-samples",
                        required=False, 
                        default=2048,
                        type=int,
                        help="Number of samples for adaround.")
    parser.add_argument("--adaround-iterations",
                        required=False, 
                        default=20000,
                        type=int)
    parser.add_argument("--seed",
                        required=False, 
                        type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument("--adaround",
                        action="store_true")
    parser.add_argument("--num-workers",
                        type=int,
                        required=False,
                        default=4)
    
    args = parser.parse_args()
    if args.qscheme not in ['tf_enhanced', 'tf']:
        raise ValueError(f'Unrecognised qscheme: {args.qscheme}')
    
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
        
    # setting random model name for adaround tmp files 
    ada_path = f'{args.model_name}_{time.time()}'
    
    if args.model_name == 'resnet18':
        model = resnet18(pretrained=True)
    elif args.model_name == 'resnet34':
        model = resnet34(pretrained=True)
    elif args.model_name == 'resnet50':
        model = resnet50(pretrained=True)
    elif args.model_name == 'deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    elif args.model_name == 'unet':
        model = model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    else:
        raise ValueError('Unrecognized model name')
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
    
    # BOPs = output_bw * MACs * param_bw
    bops = args.param_bw / 32 * args.output_bw / 32 * 100
    logging.info(f'%BOPS: {bops}')
    if run: wandb.log({'bops': bops})
        
    # quantization    
    model = prepare_model(model)
    _ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
    dummy_input = torch.rand(1, 3, 224, 224) 
    dummy_input = dummy_input.to(device)
    
    if args.qscheme == 'tf_enhanced':
        qscheme = QuantScheme.post_training_tf_enhanced
    else:
        qscheme = QuantScheme.post_training_tf
        
    if args.adaround:
        params = AdaroundParameters(data_loader=val_loader, 
                                    num_batches=args.adaround_samples//val_loader.batch_size, 
                                    default_num_iterations=args.adaround_iterations)
        os.makedirs('adaround/', exist_ok=True)
        model = Adaround.apply_adaround(model, dummy_input, params,
                                        path='adaround/', 
                                        filename_prefix=ada_path, 
                                        default_param_bw=args.param_bw,
                                        default_quant_scheme=qscheme)
        
    sim = QuantizationSimModel(model=model,
                               quant_scheme=qscheme,
                               dummy_input=dummy_input,
                               default_output_bw=args.output_bw,
                               default_param_bw=args.param_bw)
    logging.info(str(sim))
#     logging.info(str(sim.__dir__))
    
    if args.adaround:
        sim.set_and_freeze_param_encodings(
            encoding_path=os.path.join('adaround/', f'{ada_path}.encodings'))
    
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