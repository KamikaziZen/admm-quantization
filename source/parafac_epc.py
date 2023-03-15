import numpy as np
import torch
import tensorly as tl

from tensorly.decomposition import parafac
from tensorly.decomposition.candecomp_parafac import initialize_factors
from tensorly.kruskal_tensor import KruskalTensor, kruskal_to_tensor

from musco.pytorch.compressor.decompose.cpd.lib_anc import cp_anc


def parafac_epc(tensor, rank, als_maxiter=5000, als_tol=1e-5, num_threads=4,
                init="random", epc_maxiter=5000, epc_rounds=50, epc_tol=1e-5,
                stop_tol=1e-4, ratio_tol=1e-3, ratio_max_iters=10):
    """
    Factorize convolutional layer using CP decomposition, with EPC (optionally)
    Parameters:
    tensor:          N-way tensor to be decomposed, tensor or ndarray
    rank:            Rank of CP decomposition, int
    init:            Way to initialize CP decomposition, "random" or "svd"
    als_maxiter:     Number of ALS iterations, int
    epc_maxiter:     Number of EPC iterations in one round, int
    epc_rounds:      Number of EPC rounds, int
    als_tol:         Tolerance of Fast ALS algorithm, float
    epc_tol:         Tolerance of EPC algorithm, float
    stop_tol:        Tolerance of intensity criteria, float
    ratio_tol:       Tolerance of ratio criteria, float
    ratio_max_iters: How many time to satisfy stopping criteria to finish, int
    num_threads:     Number of threads to perform compression
    Output:
    lmbda, Us:       Lambda, List of factors
    """
    torch.set_num_threads(num_threads)
    tl.set_backend('pytorch')
    
    Y = torch.as_tensor(tensor, dtype=torch.float64, device=tensor.device)
    shape = Y.size()
    order = np.argsort(shape)
    iorder = np.argsort(order)[::-1]
    Y = Y.permute(tuple(order))
    
    P_init = parafac(Y, rank, n_iter_max=als_maxiter, init=init,
                     tol=als_tol, svd=None, normalize_factors=True)

    Us = [torch.tensor(P_init.factors[i]) for i in iorder]
    print(Us[0].dtype)
    # Us[-1] *= P_init.weights
    lmbda = P_init.weights

    # Apply EPC
    delta = torch.norm(Y - kruskal_to_tensor(P_init))
    norm_lda0 = torch.norm(P_init.weights)
    P_init.factors[-1] *= P_init.weights
    P_als = KruskalTensor((None, P_init.factors))

    lambda_norm_prev = norm_lda0
    alpha_prev = P_init.weights.max() / P_init.weights.min()
    alpha_list = [alpha_prev]

    stopflag = 0
    for i in range(epc_rounds):
        print(f'{i} round')
        P_als = cp_anc(Y, rank, delta, P_init=P_als, maxiter=epc_maxiter, tol=epc_tol)
        lambda_norm = torch.norm(P_als.weights)
        alpha = P_als.weights.max() / P_als.weights.min()
        # check that sum of squares of intensities isn't change much
        if torch.abs(lambda_norm_prev - lambda_norm) < stop_tol * lambda_norm_prev: break
        # check ratio between max and min intensities
        stopflag = stopflag + 1 if torch.abs(alpha_prev - alpha) < ratio_tol else 0
        lambda_norm_prev = lambda_norm
        alpha_prev = alpha
        alpha_list += [alpha]

        if stopflag >= ratio_max_iters: break


    Us = [torch.tensor(P_als.factors[i]) for i in iorder[::-1]]  # (f_cin, f_cout, f_z)
    # Us[-1] *= P_als.weights
    lmbda = P_als.weights
    print(Us[0].dtype)

    return lmbda, Us