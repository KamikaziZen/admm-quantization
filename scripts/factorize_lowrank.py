from functools import partial
import torch
import numpy as np
import random
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from source.quantization import quantize_tensor


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def run_name(args):
    run_name = []
    run_name.append(f"b={args.bits}")
    run_name.append(f"r={args.rank}")
    run_name.append(f"s={args.seed}")
    
    return '_'.join(run_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-id",
                        type=str,
                        help="[huggyllama/llama-7b, ]")
    parser.add_argument("--cache-dir",
                        type=str,
                        help="Cache directory for huggingface models and datasets")
    parser.add_argument("--output-dir",
                        type=str,
                        help="Output dir for saving factors.")
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="Whether to enable experiment logging to wandb.")
    parser.add_argument("--layer",
                        type=str,
                        help="Layer to be ADMMed.")
    parser.add_argument("--max-iter",
                        required=True, 
                        type=int,
                        default=100,
                        help="Max number of iterations of admm.")
    parser.add_argument("--bits",
                        required=True, 
                        type=int,
                        default=4,
                        help="Bit-width of quantization.")
    parser.add_argument("--rank",
                        required=True, 
                        type=int,
                        default=4,
                        help="Rank of lora component.")
    parser.add_argument("--qscheme",
                        required=False,
                        default='tensor_minmax',
                        type=str,
                        help='Use tf for usual min-max and tf_enhanced for sqnr-optimized min-max.')
    parser.add_argument("--seed",
                        required=False, 
                        type=int,
                        default=42,
                        help="Random seed.")
    
    args = parser.parse_args()
    # if args.qscheme not in ['tf_enhanced', 'tf']:
    #     raise ValueError(f'Unrecognised qscheme: {args.qscheme}')
    
    return args


def project_rank(H, rank):
    U, S, Vt = torch.linalg.svd(H)
    return U[:,:rank] @ torch.diag(S[:rank]) @ Vt[:rank]


def admm_iteration(H, U, W, H2, proj_func, rho=1.0, max_iter=50, eps=1e-8):
    # rank = H.shape[1]
    # rho = torch.trace(G) / rank
    # rho = 1.0
    for j in range(1, max_iter):
        H_ = (rho*(H + U) + W - H2) / (1 + rho)
        H_prev = H.clone()
        # H = quantize_tensor(H_T - U, qscheme=qscheme, bits=bits)
        H = proj_func(H_ - U)
        U += H - H_

        r = torch.sum((H - H_)**2) / torch.sum(H**2)
        s = torch.sum((H - H_prev)**2) / torch.sum(U**2) 
        if r < eps and s < eps: 
            break
            
    return H, U


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
    
    model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                 device_map={"":0},
                                                 cache_dir=args.cache_dir)
    state_dict = model.state_dict()

    init = 'random'
    quantize_func = partial(quantize_tensor, qscheme=args.qscheme, bits=args.bits)
    project_func = partial(project_rank, rank=args.rank)
    
    # original matrix
    # W = state_dict['model.layers.0.self_attn.q_proj.weight']
    W = state_dict[f'{args.layer}.weight']
    
    # init W_q as quantized W or as random matrix
    if init == 'random':
        W_q = torch.randn(*W.shape, device=device)
    else:
        W_q = quantize_func(W)
    U_q = torch.zeros_like(W_q, device=device)
    
    # init W_r as low-rank random matrix
    W_r = torch.randn(*W.shape, device=device)
    W_r = project_func(W_r)
    U_r = torch.zeros_like(W_r, device=device)
    
    logging.info(f'Bits: {args.bits}, Rank: {args.rank}')
    abs_quant_diff = torch.linalg.norm(W - quantize_func(W))
    rel_quant_diff = abs_quant_diff / torch.linalg.norm(W)
    logging.info(f'Diff between W and quantized W abs: {abs_quant_diff:.4f}, rel: {rel_quant_diff:.4f}')
    if run: wandb.log({'abs_quant_diff': abs_quant_diff, 'rel_quant_diff': rel_quant_diff})
    
    # ADMM iteration
    rel_admm_diff = None
    prev_rel_admm_diff = None
    rho = 1.0
    for i in range(args.max_iter):
        W_q, U_q = admm_iteration(W_q, U_q, W, W_r, quantize_func, rho=rho)
        W_r, U_r = admm_iteration(W_r, U_r, W, W_q, project_func, rho=rho)

        if rel_admm_diff is not None:
            prev_rel_admm_diff = rel_admm_diff
        abs_admm_diff = torch.linalg.norm(W - W_r - W_q)
        rel_admm_diff = abs_admm_diff /torch.linalg.norm(W)
        
        if i % 10 == 0:
            logging.info(f'Diff between W and (W_q + W_r) abs: {abs_admm_diff:.4f}, rel: {rel_admm_diff:.4f}')
            if run: 
                wandb.log({'abs_admm_diff': abs_admm_diff, 'rel_admm_diff': rel_admm_diff})

        if prev_rel_admm_diff and prev_rel_admm_diff < rel_admm_diff - 1: break

    torch.save(W_q, 
               os.path.join(args.output_dir,
                            f'{args.layer}_{args.bits}_{args.rank}_{rel_admm_diff:.3f}_Q.pt'))
    torch.save(W_r, 
               os.path.join(args.output_dir,
                            f'{args.layer}_{args.bits}_{args.rank}_{rel_admm_diff:.3f}_R.pt'))

    if run: run.finish()


if __name__ == '__main__':
    main()