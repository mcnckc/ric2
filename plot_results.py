import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os
from scipy.io import mmread
from scipy.sparse.linalg import cg, spilu, LinearOperator
from ric2cg import RIC2CG

RESULTS_FILE = 'benchmark_results.csv'
OUTPUT_DIR = 'graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Could not find {RESULTS_FILE}. Run benchmarks first!")
    
    df = pd.read_csv(RESULTS_FILE)
    
    # 1. IDENTIFY METHODS
    method_cols = [c for c in df.columns if c.endswith('_Iter')]
    methods = [c.replace('_Iter', '') for c in method_cols]
    print(methods)
    print(f"Found data for methods: {methods}")
    
    long_data = []
    
    for i, row in df.iterrows():
        base_nnz = row['nnz']
        
        for m in methods:
            iter_val = row.get(f"{m}_Iter", np.nan)
            time_val = row.get(f"{m}_Time", np.nan)
            p_nnz = row.get(f"{m}_NNZ", np.nan)
            
            if pd.isna(p_nnz):
                fill_factor = 0 
                if m == 'Jacobi': fill_factor = row['rows'] / base_nnz 
            else:
                fill_factor = p_nnz / base_nnz
                
            if not pd.isna(iter_val):
                long_data.append({
                    'Matrix': row['name'],
                    'Method': m,
                    'Iterations': iter_val,
                    'Time': time_val,
                    'Fill_Factor': fill_factor,
                    'Matrix_NNZ': base_nnz,
                    'Success': True 
                })
            else:
                 long_data.append({
                    'Method': m,
                    'Success': False
                })
                
    return pd.DataFrame(long_data), df

def plot_pareto_frontier(long_df):
    print("Generating Plot 1: Pareto Frontier...")
    
    plot_df = long_df[long_df['Success'] == True].copy()
    plot_df = plot_df[~plot_df['Method'].str.contains('Exact')]
    plot_df = plot_df[~plot_df['Method'].str.contains('CG')] 
    print(plot_df)
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=plot_df, 
        x='Fill_Factor', 
        y='Iterations', 
        hue='Method',
        style='Method',
        s=100, 
        alpha=0.8
    )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Pareto Frontier: Memory vs. Convergence\n(Lower Left is Better)", fontsize=14)
    plt.xlabel("Fill-in Factor (Preconditioner NNZ / A NNZ)", fontsize=12)
    plt.ylabel("Iterations to Convergence (Log Scale)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot1_pareto_frontier.png")
    plt.close()

def plot_robustness(long_df):
    print("Generating Plot 3: Robustness...")
    
    counts = long_df.groupby('Method')['Success'].value_counts().unstack().fillna(0)
    
    if True in counts.columns:
        counts['Total'] = counts.sum(axis=1)
        counts['Success_Rate'] = (counts[True] / counts['Total']) * 100
    else:
        counts['Success_Rate'] = 0

    counts = counts.sort_values('Success_Rate', ascending=False).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x='Success_Rate', y='Method', palette='viridis')
    plt.title("Robustness: Success Rate by Method", fontsize=14)
    plt.xlabel("Percentage of Matrices Converged (%)", fontsize=12)
    plt.xlim(0, 105)
    plt.axvline(100, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot3_robustness.png")
    plt.close()

def run_trace_for_plot(A, b,  method_name, x0=None, **kwargs):
    """Helper to run a solver and capture FULL history for Plot 2"""
    history = []
    bnorm = np.linalg.norm(b) or 1.0

    def cb(xk):
        # Robust to SciPy callback variants
        if np.isscalar(xk):
            # If SciPy ever gives a norm-like value, at least record it.
            history.append(float(xk) / bnorm)
        else:
            res = np.linalg.norm(A @ xk - b) / bnorm
            history.append(res)

    N = A.shape[0]

    if method_name == 'CG':
        try:
            cg(A, b, x0=x0, callback=cb, callback_type='x', rtol=1e-6, maxiter=2000)
        except TypeError:
            cg(A, b, x0=x0, callback=cb, rtol=1e-6, maxiter=2000)

    elif method_name == 'Jacobi':
        d = np.array(A.diagonal(), copy=True)
        d[d == 0] = 1
        M = sp.diags(1 / d)
        try:
            cg(A, b, x0=x0, M=M, callback=cb, callback_type='x', rtol=1e-6, maxiter=2000)
        except TypeError:
            cg(A, b, x0=x0, M=M, callback=cb, rtol=1e-6, maxiter=2000)

    elif 'ILU' in method_name:
        tol = kwargs.get('tol', 1e-4)
        fill_factor = kwargs.get('fill_factor', 100)

        # Ensure spilu gets a CSC sparse matrix
        A_csc = A if sp.isspmatrix(A) else sp.csc_matrix(A)
        A_csc = A_csc.tocsc()

        M = None
        try:
            ilu = spilu(A_csc, drop_tol=tol, fill_factor=fill_factor)
            print("ILU NONZEROS:", ilu.L.nnz + ilu.U.nnz, ilu.nnz)
            M = LinearOperator((N, N), matvec=ilu.solve)

        except Exception as e:

            print("SPILU FAILS", e)
            # ILU failed -> we'll fall back to plain CG below
            M = None

        try:
            x, n_iter = cg(A_csc, b, x0=x0, M=M, callback=cb, rtol=1e-6, maxiter=2000)
            print("ILU FINE", n_iter)
        except TypeError as e:
            x, n_iter = cg(A_csc, b, x0=x0, M=M, callback=cb, rtol=1e-6, maxiter=2000)
            print(f"ILU exception{e}, n_iter", n_iter)

    elif 'RIC2S' in method_name:
        tau = kwargs.get('tau', 0.1)
        _, hist, nnz = RIC2CG.solve(A, b, x0=x0, stable=True, tau=tau, rtol=1e-6, maxiter=2000)
        print("RIC2S NNZ", nnz)
        history = hist

    return history


# def run_trace_for_plot(A, b, method_name, **kwargs):
#     """Helper to run a solver and capture FULL history for Plot 2"""
#     history = []
#     def cb(xk):
#         res = np.linalg.norm(A @ xk - b) / np.linalg.norm(b)
#         history.append(res)
    
#     N = A.shape[0]
    
#     if method_name == 'CG':
#         cg(A, b, callback=cb, rtol=1e-6, maxiter=2000)
#     elif method_name == 'Jacobi':
#         d = A.diagonal()
#         d[d==0]=1
#         M = sp.diags(1/d)
#         cg(A, b, M=M, callback=cb, rtol=1e-6, maxiter=2000)
#     elif 'ILU' in method_name:
#         tol = kwargs.get('tol', 1e-4)
#         try:
#             ilu = spilu(A.tocsc(), drop_tol=tol, fill_factor=10)
#             M = LinearOperator((N,N), matvec=ilu.solve)
#             cg(A, b, M=M, callback=cb, rtol=1e-6, maxiter=2000)
#         except: pass
#     elif 'RIC2S' in method_name:
#         tau = kwargs.get('tau', 0.1)
#         _, hist, _ = RIC2CG.solve(A, b, stable=True, tau=tau, rtol=1e-6, maxiter=2000)
#         history = hist

#     return history

def plot_convergence_history(original_df):
    print("Generating Plot 2: Convergence History (Re-running solvers)...")
    
    available_df = original_df[original_df['path'].apply(os.path.exists)].sort_values('rows', ascending=False)
    if available_df.empty:
        print("No matrices available for history plot.")
        return

    target_row = available_df.iloc[5]
    name = target_row['name']
    path = target_row['path']
    print(f"  > Selected Matrix: {name} (Rows: {target_row['rows']})")
    
    try:
        A = mmread(path).tocsr()
        N = A.shape[0]
        print("A size", N, "NNZ", A.nnz)
        b = np.random.randn(N)
        x0 = np.random.randn(N)
        traces = {}
        
        #traces['CG'] = run_trace_for_plot(A, b, 'CG')
        #traces['Jacobi'] = run_trace_for_plot(A, b, 'Jacobi')
        traces['ILU (1e-4)'] = run_trace_for_plot(A, b, 'ILU', tol=1e-4, fill_factor=5)
        traces['RIC2S (tau=0.001)'] = run_trace_for_plot(A, b, 'RIC2S', tau=1e-2)
        
        plt.figure(figsize=(10, 6))
        
        for label, history in traces.items():
            print(label, history[0])
            if not history: continue
            plt.plot(history, label=label, linewidth=2)
            
        plt.yscale('log')
        plt.title(f"Convergence History: {name}", fontsize=14)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Relative Residual (Log10)", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plot2_convergence_history.png")
        plt.close()
        
    except Exception as e:
        print(f"Could not generate history plot: {e}")

def main():
    print("Loading Results...")
    long_df, original_df = load_and_preprocess()
    
    plot_pareto_frontier(long_df)
    #plot_robustness(long_df)
    #plot_convergence_history(original_df)
    
    print(f"\nAll plots saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()