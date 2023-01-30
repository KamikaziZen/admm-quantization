import numpy as np
import torch
import tensorly as tl

from tensorly.decomposition import parafac
from tensorly.decomposition.candecomp_parafac import initialize_factors
from tensorly.kruskal_tensor import KruskalTensor, kruskal_to_tensor

from musco.pytorch.compressor.decompose.cpd.lib_fastals import cp_fast_als
from musco.pytorch.compressor.decompose.cpd.lib_anc import cp_anc
from musco.pytorch.compressor.decompose.cpd.lib import backend, to_tensor, B, Bla, save, size, transpose, copy_tensor


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
    
    Y = to_tensor(tensor, dtype=B.float64)
    shape = size(Y)
#     print('Y shape:', shape)
    order = np.argsort(shape)
#     print('order:', order)
    iorder = np.argsort(order)[::-1]
#     print('iorder:', iorder)
    Y = transpose(Y, order)
    
    tl.set_backend('numpy')
    P_init = parafac(Y, rank, n_iter_max=als_maxiter, init=init,
                     tol=als_tol, svd=None, normalize_factors=True)

    Us = [copy_tensor(P_init.factors[i]) for i in iorder]
#     print([factor.shape for factor in Us])
#     Us = [copy_tensor(factor) for factor in P_init.factors]
    # Us[-1] *= P_init.weights
    lmbda = P_init.weights

    # Apply EPC
    delta = Bla.norm(Y - kruskal_to_tensor(P_init))
    norm_lda0 = Bla.norm(P_init.weights)
    P_init.factors[-1] *= P_init.weights
    P_als = KruskalTensor((None, P_init.factors))

    lambda_norm_prev = norm_lda0
    alpha_prev = P_init.weights.max() / P_init.weights.min()
    alpha_list = [alpha_prev]

    stopflag = 0
    for i in range(epc_rounds):
        print(f'{i} round')
        P_als = cp_anc(Y, rank, delta, P_init=P_als, maxiter=epc_maxiter, tol=epc_tol)
        lambda_norm = Bla.norm(P_als.weights)
        alpha = P_als.weights.max() / P_als.weights.min()
        # check that sum of squares of intensities isn't change much
        if B.abs(lambda_norm_prev - lambda_norm) < stop_tol * lambda_norm_prev: break
        # check ratio between max and min intensities
        stopflag = stopflag + 1 if B.abs(alpha_prev - alpha) < ratio_tol else 0
        lambda_norm_prev = lambda_norm
        alpha_prev = alpha
        alpha_list += [alpha]
        # save(P_als, self.template.format("tmp"))

        if stopflag >= ratio_max_iters: break

    # save(alpha_list, self.template.format("lam"))

    # save factors
    Us = [copy_tensor(P_als.factors[i]) for i in iorder[::-1]]  # (f_cin, f_cout, f_z)
#     print([factor.shape for factor in Us])
#     Us = [copy_tensor(factor) for factor in P_als.factors]
    # Us[-1] *= P_als.weights
    lmbda = P_als.weights

    return lmbda, Us