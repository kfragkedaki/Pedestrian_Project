import numpy as np
import pypolycontain as pp
from typing import List
from src.reachability_analysis.operations import *


def LTI_reachability(
    U_minus: np.ndarray,
    X_plus: np.ndarray,
    X_minus: np.ndarray,
    X_0: pp.zonotope,
    Z_w: pp.zonotope,
    M_w: pp.zonotope,
    U_k: Union[pp.zonotope, List[pp.zonotope]],
    N: int = 30,
    n: int = 50,
    disable_progress_bar: bool = False,
) -> List[pp.zonotope]:
    """Linear time-invariant reachability analysis

    Parameters:
    -----------
    U_minus : np.ndarray
        Input array of all inputs in each trajectory, looking like
        [u(1)(0) . . . u(1)(T1-1) . . . u(K)(0) . . . u(K)(TK-1)]
    X_plus : np.ndarray
        State array of all states in each trajectory from T=1
        (instead of T=0 as start), on the form
        [x(1)(1) . . . x(1)(T1) . . . x(K)(1) . . . x(K)(TK )]
    X_minus : np.ndarray
        State array of all states in each trajectory to T-1, on the form
        [x(1)(0) . . . x(1)(T1-1) . . . x(K)(0) . . . x(K)(TK-1)]
    X_0 : pp.zonotope
        Initial state represented as a zonotope
    Z_w : pp.zonotope
        Process noise zonotope
    M_w : pp.zonotope
        Concatenation of multiple noise zonotopes
    U_k : pp.zonotope
        Input zonotope
    N : int (default = 30)
        Number of timesteps
    n : int (default = 100)
        Reduced order of reachable sets
    disable_progress_bar : bool (default = False)
        Disables the progress bar
    """
    if type(U_k) == pp.zonotope:
        U_k = [U_k] * N
    R = [0] * N
    R[0] = reduce(X_0, order=n)
    _stacked = np.vstack([X_minus, U_minus])
    _X = matrix_zonotope(X_plus - M_w.x, M_w.G)
    M_sigma = product(_X, np.linalg.pinv(_stacked))
    for i in tqdm(
        range(0, N - 1), desc="Calculating reachable sets", disable=disable_progress_bar
    ):
        R[i + 1] = minkowski_sum(product(M_sigma, cartesian_product(R[i], U_k[i])), Z_w)
        R[i + 1] = reduce(R[i + 1], order=n)
    return R
