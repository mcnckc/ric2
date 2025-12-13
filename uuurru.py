import scipy.sparse as sp
import numpy as np

def RIC2(A, tau=0.1):
    """
    Calculate sparse U, R factors in U^TU+U^TR+R^TU decomposition of sparse symmetric positive definite matrix A
    Smaller tau makes U less sparse, but makes ||R|| smaller, tau=0 gives cholesky (R=0)
    Fill-in amount in R is not controlled, so complexity >= O(n*nnz(A))
    """
    A = A.tocsr()
    n = A.shape[0]
    U, Y, R, Z = sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n))
    gamma_k = np.sqrt(A[0, 0])
    vk = A[0:1, 1:] / gamma_k
    zk = vk.copy()
    yk = vk.copy()
    yk.data[np.abs(vk.data) < tau] = 0
    zk.data[np.abs(vk.data) >= tau] = 0
    yk.eliminate_zeros()
    zk.eliminate_zeros()
    U[0, 0] = gamma_k
    Y[0, 1:] = yk
    Z[0, 1:] = zk
    for i in range(1, n):
        Yk = Y[:i, i:]
        Zk = Z[:i, i:]
        uk = Yk[:, 0:1]
        Yk = Yk[:, 1:]
        rk = Zk[:, 0:1]
        Zk = Zk[:, 1:]
       
        gamma_k = np.sqrt(A[i, i] - (uk.multiply(uk)).sum())

        U[:i, i] = uk
        U[i, i] = gamma_k
        R[:i, i] = rk
        if i + 1 < n:
            ak = A[i:i+1, i + 1:]
            vk = (ak - uk.T @ Yk - uk.T @ Zk - rk.T @ Yk) / gamma_k
            zk = vk.copy()
            yk = vk.copy()
            yk.data[np.abs(vk.data) < tau] = 0
            zk.data[np.abs(vk.data) >= tau] = 0
            yk.eliminate_zeros()
            zk.eliminate_zeros()
            
            Y[i, i + 1:] = yk
            Z[i, i + 1:] = zk
        
    return U, R