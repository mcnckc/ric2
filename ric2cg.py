import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve_triangular

class RIC2CG:
    @staticmethod
    def solve(A, b, x0=None, stable=False, tau=None, rtol=1e-6, maxiter=5000):
        """
        Solve Ax=b using RIC2 preconditioner
        A is scipy sparse SPD matrix
        b is numpy vector
        stable=True means to use RIC2S version (more stable, less fill-in, faster 
        to compute decomposition. May need more CG iterations)
        tau is a RIC2 hyperparameter
        RETURNS:
        x - solution vector(numpy array)
        residuals - norms ||Ax-b||/||b|| after each iteration
        nnz - number of nonzeros in preconditioner
        """
        n = A.shape[0]
        D = 1 / np.sqrt(A.diagonal())
        A, b = sp.diags(D) @ A.copy() @ sp.diags(D), D * b

        if stable:
            U, R = RIC2CG._RIC2S(A, tau)
        else:
            U, R = RIC2CG._RIC2(A, tau)

        residuals = []
        def solveUTU(x):
            y = spsolve_triangular(U.T, x, lower=True)
            return spsolve_triangular(U, y, lower=False)
        
        norm_b = np.linalg.norm(b / D)
        def save_x(x):
            residuals.append(np.linalg.norm((A @ x - b) / D) / norm_b)

        xf, n_iters = sp.linalg.cg(A, b, x0=x0, M=sp.linalg.LinearOperator(shape=(n, n), matvec=solveUTU, dtype=np.double), 
                     rtol=1e-9, maxiter=maxiter, callback=save_x)

        return xf * D, residuals, 2 * U.nnz
        
    @staticmethod
    def solvelr(A, b, x0=None, stable=False, tau=None, rtol=1e-6, maxiter=5000):
        """
        Solve Ax=b using RIC2 preconditioner
        A is scipy sparse SPD matrix
        b is numpy vector
        stable=True means to use RIC2S version (more stable, less fill-in, faster 
        to compute decomposition. May need more CG iterations)
        tau is a RIC2 hyperparameter
        RETURNS:
        x - solution vector(numpy array)
        residuals - norms ||Ax-b||/||b|| after each iteration
        nnz - number of nonzeros in preconditioner
        """
        n = A.shape[0]
        D = 1 / np.sqrt(A.diagonal())
        A, b = sp.diags(D) @ A.copy() @ sp.diags(D), D * b

        if stable:
            U, R = RIC2CG._RIC2S(A, tau)
        else:
            U, R = RIC2CG._RIC2(A, tau)

        def Ax(x):
            y = spsolve_triangular(U, x, lower=False)
            y = A @ y
            return spsolve_triangular(U.T, y, lower=True)
        Aop = sp.linalg.LinearOperator(shape=(n, n), matvec=Ax, dtype=np.double)
        b = spsolve_triangular(U.T, b, lower=True)

        residuals = []
        
        norm_b = np.linalg.norm((U.T @ b) / D)
        def save_x(x):
            residuals.append(np.linalg.norm((U.T @ (Ax(x) - b)) / D) / norm_b)

        xf, n_iters = sp.linalg.cg(Aop, b, x0=x0, 
                     rtol=1e-9, maxiter=maxiter, callback=save_x)

        return spsolve_triangular(U, xf, lower=False) * D, residuals, 2 * U.nnz
    
    @staticmethod
    def _RIC2(A, tau=None):
        """
        Calculate sparse U, R factors in U^TU+U^TR+R^TU decomposition of sparse symmetric positive definite
        UNIDIAGONAL matrix A
        Smaller tau makes U less sparse, but makes ||R|| smaller, tau=0 gives cholesky (R=0)
        Fill-in amount in R is not controlled, so complexity >= O(n*nnz(A))
        """
        if tau is None:
            tau = 0.1
        A = A.tocsr()
        n = A.shape[0]
        # offtop : we can try to use the faster data formats : else it would be just oyhton loops
        U, Y, R, Z = sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n))
        for i in range(0, n):
            gamma_k = A[i, i]
            if i > 0:
                Yk = Y[:i, i:]
                Zk = Z[:i, i:]
                uk = Yk[:, 0:1]
                Yk = Yk[:, 1:]
                rk = Zk[:, 0:1]
                Zk = Zk[:, 1:]

                U[:i, i] = uk
                R[:i, i] = rk
                
                gamma_k -= ((uk.multiply(uk)).sum())
            gamma_k = np.sqrt(gamma_k)
        
            U[i, i] = gamma_k
            
            if i + 1 < n:
                vk = A[i:i+1, i + 1:]
                if i > 0:
                    vk -= (uk.T @ Yk + uk.T @ Zk + rk.T @ Yk)
                vk /= gamma_k
                zk = vk.copy()
                yk = vk.copy()
                yk.data[np.abs(vk.data) < tau] = 0
                zk.data[np.abs(vk.data) >= tau] = 0
                yk.eliminate_zeros()
                zk.eliminate_zeros()
                
                Y[i, i + 1:] = yk
                Z[i, i + 1:] = zk
            
        return U, R
    @staticmethod
    def _RIC2S(A, tau=None):
        """
        Calculate sparse U, R factors in U^TU+U^TR+R^TU+S decomposition of sparse symmetric positive definite
        UNIDIAGONAL matrix A
        Smaller tau makes U less sparse, but makes ||R|| smaller, tau=0 gives cholesky (R=0)
        Fill-in amount in R is more stabilized
        Eiagenvalues of U^-TAU^-1 become more stable
        """
        A = A.tocsr()
        n = A.shape[0]

        if tau is None:
            tau = np.sqrt(n / A.nnz)
        
        print(A.shape)
        U, Y, R, Z = sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n)), sp.csc_array((n, n))
        d = (1 + 2 * tau ** 2) * A.diagonal()
        for i in range(0, n):
            if i > 0:
                Yk = Y[:i, i:]
                Zk = Z[:i, i:]
                uk = Yk[:, 0:1]
                Yk = Yk[:, 1:]
                rk = Zk[:, 0:1]
                Zk = Zk[:, 1:]
                
                U[:i, i] = uk
                R[:i, i] = rk

            if i + 1 < n:
                vk = A[i:i+1, i + 1:]
                if i > 0:
                    vk -= (uk.T @ Yk + uk.T @ Zk + rk.T @ Yk)
                # we loop is slow???     
                dz = np.abs(vk.copy()).multiply(sp.csr_array(1 / np.sqrt(d[None, 1:])))
                for j, dzj in zip(dz.indices, dz.data):
                    if dzj <= tau ** 2:
                        d[0] *= (1 + dzj)
                        d[j + 1] *= (1 + dzj)
                        vk[0, j] = 0
                vk /= np.sqrt(d[0])
                
                zk = vk.copy()
                yk = vk.copy()
                yk.data[np.abs(vk.data) < tau] = 0
                zk.data[np.abs(vk.data) >= tau] = 0
                yk.eliminate_zeros()
                zk.eliminate_zeros()
                
                Y[i, i + 1:] = yk
                Z[i, i + 1:] = zk
                d[1:] -= yk.power(2)
            U[i, i] = np.sqrt(d[0])
            d = d[1:]
            
        return U, R