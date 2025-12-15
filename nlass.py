import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spilu, splu, LinearOperator, spsolve_triangular
from ric2cg import RIC2CG

class NLASS:
    def __init__(self, A):
        self.A = A.tocsr()
        self.n = A.shape[0]
        self.b = np.random.randn(self.n)

    def _calculate_residual(self, x):
        norm_b = np.linalg.norm(self.b)
        if norm_b == 0: return 0.0
        r = self.b - self.A @ x
        return np.linalg.norm(r) / norm_b

    def _run_solver(self, method_func, name, nnz_precond=0, is_custom=False, **kwargs):
        start = time.time()
        iters = 0
        final_res = 0.0
        
        try:
            if is_custom:
                x, residuals, p_nnz = method_func(**kwargs)
                iters = len(residuals)
                final_res = residuals[-1] if residuals else self._calculate_residual(x)
                nnz_precond = p_nnz + self.A.nnz
            else:
                def callback(xk):
                    nonlocal iters
                    iters += 1
                
                if name == "Exact":
                    x = method_func()
                    iters = 1
                    final_res = 0.0
                else:
                    x, info = method_func(callback=callback, **kwargs)
                    final_res = self._calculate_residual(x)
                
        except Exception:
            return {
                f"{name}_Time": np.nan, 
                f"{name}_Iter": np.nan, 
                f"{name}_Res": np.nan,
                f"{name}_NNZ": np.nan
            }
        
        end = time.time()
        
        return {
            f"{name}_Time": end - start,
            f"{name}_Iter": iters,
            f"{name}_Res": final_res,
            f"{name}_NNZ": nnz_precond
        }

    def test_cg_none(self):
        func = lambda callback, **kwargs: cg(self.A, self.b, rtol=1e-6, maxiter=5000, callback=callback)
        return self._run_solver(func, "CG", nnz_precond=self.A.nnz)

    def test_cg_jacobi(self):
        diags = self.A.diagonal()
        diags[diags == 0] = 1.0 
        M_x = lambda x: x / diags
        M = LinearOperator((self.n, self.n), matvec=M_x)
        
        func = lambda callback, **kwargs: cg(self.A, self.b, M=M, rtol=1e-6, maxiter=5000, callback=callback)
        return self._run_solver(func, "Jacobi", nnz_precond=self.n + self.A.nnz)

    def test_cg_ilu(self, drop_tol=1e-4):
        name = f"ILU_{drop_tol}"
        try:
            ilu = spilu(self.A.tocsc(), drop_tol=drop_tol, fill_factor=20)
            M_x = ilu.solve
            M = LinearOperator((self.n, self.n), matvec=M_x)
            p_nnz = ilu.nnz
            
            func = lambda callback, **kwargs: cg(self.A, self.b, M=M, rtol=1e-6, maxiter=5000, callback=callback)
            return self._run_solver(func, name, nnz_precond=p_nnz + self.A.nnz)
        except Exception:
            return {f"{name}_Time": np.nan, f"{name}_Iter": np.nan, f"{name}_Res": np.nan, f"{name}_NNZ": np.nan}
    def test_cg_sgs(self):
        L_tri = sp.tril(self.A, format='csr') 
        U_tri = sp.triu(self.A, format='csr')
        diag = self.A.diagonal()
        def sgs_matvec(r):
            y = spsolve_triangular(L_tri, r, lower=True)
            y = y * diag
            z = spsolve_triangular(U_tri, y, lower=False)
            return z
        M = LinearOperator((self.n, self.n), matvec=sgs_matvec)
        p_nnz = self.A.nnz
        func = lambda callback, **kwargs: cg(self.A, self.b, M=M, rtol=1e-6, maxiter=5000, callback=callback)
        return self._run_solver(func, "SGS", nnz_precond=p_nnz + self.A.nnz)
    def test_exact(self):
        try:
            lu = splu(self.A.tocsc())
            p_nnz = lu.nnz
            def run_direct():
                return lu.solve(self.b)
            return self._run_solver(run_direct, "Exact", nnz_precond=p_nnz + self.A.nnz)
        except Exception:
            print("Exact LU Failed!s")
            return {"Exact_Time": np.nan, "Exact_Iter": np.nan, "Exact_Res": np.nan, "Exact_NNZ": np.nan}

    def test_ric2(self, stable=False, tau=None):
        mode = "RIC2S" if stable else "RIC2"
        tau_str = str(tau) if tau else "Auto"
        name = f"{mode}_tau{tau_str}"

        func = lambda: RIC2CG.solve(self.A, self.b, stable=stable, tau=tau, rtol=1e-6, maxiter=5000)
        
        return self._run_solver(func, name, is_custom=True)