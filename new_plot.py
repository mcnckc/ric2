import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse as sp
from scipy.io import mmread
from scipy.sparse.linalg import cg, spilu, LinearOperator
from ric2cg import RIC2CG

RESULTS_FILE = 'benchmark_results.csv'
OUTPUT_DIR = 'graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def load_and_preprocess():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Could not find {RESULTS_FILE}. Run benchmarks first!")
    df = pd.read_csv(RESULTS_FILE)
    
    method_cols = [c for c in df.columns if c.endswith('_Iter')]
    methods = [c.replace('_Iter', '') for c in method_cols]
    
    long_data = []
    for i, row in df.iterrows():
        base_nnz = row['nnz']
        row_count = row['rows']
        for m in methods:
            iter_val = row.get(f"{m}_Iter", np.nan)
            p_nnz = row.get(f"{m}_NNZ", np.nan)
            
            if pd.isna(p_nnz):
                fill_factor = 0
                if m == 'Jacobi': fill_factor = row_count / base_nnz
            else:
                fill_factor = p_nnz / base_nnz
                
            if not pd.isna(iter_val):
                long_data.append({
                    'Matrix': row['name'], 'Rows': row_count, 'Method': m,
                    'Iterations': iter_val, 'Fill_Factor': fill_factor, 'Success': True 
                })
    return pd.DataFrame(long_data), df

def plot_pareto_frontier(long_df):
    print("Generating Plot 1: Pareto Frontier...")
    plot_df = long_df[(long_df['Success'] == True) & 
                      (~long_df['Method'].str.contains('Exact')) & 
                      (~long_df['Method'].str.contains('CG'))].copy()
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_df, x='Fill_Factor', y='Iterations', hue='Method', style='Method', s=100, alpha=0.8)
    plt.xscale('log'); plt.yscale('log')
    plt.title("Pareto Frontier: Memory vs. Convergence", fontsize=14)
    plt.xlabel("Fill-in Factor", fontsize=12); plt.ylabel("Iterations (Log Scale)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot1_pareto_frontier.png"); plt.close()

def plot_mean_performance_by_size(long_df):
    print("Generating Plot 4: Scalability...")
    plot_df = long_df[(long_df['Success'] == True) & (~long_df['Method'].str.contains('Exact'))].copy()
    bins = [0, 2000, 5000, 10000, 50000, 1000000]
    labels = ['<2k', '2k-5k', '5k-10k', '10k-50k', '>50k']
    plot_df['Size_Category'] = pd.cut(plot_df['Rows'], bins=bins, labels=labels)
    summary = plot_df.groupby(['Size_Category', 'Method'])['Iterations'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary, x='Size_Category', y='Iterations', hue='Method', palette='magma')
    plt.yscale('log')
    plt.title("Scalability: Average Iterations by Matrix Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot4_scalability_by_size.png"); plt.close()

def plot_robustness(long_df):
    print("Generating Plot 3: Robustness...")
    counts = long_df.groupby('Method')['Success'].value_counts().unstack().fillna(0)
    if True in counts.columns:
        counts['Total'] = counts.sum(axis=1)
        counts['Success_Rate'] = (counts[True] / counts['Total']) * 100
    else: counts['Success_Rate'] = 0
    counts = counts.sort_values('Success_Rate', ascending=False).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x='Success_Rate', y='Method', palette='viridis')
    plt.title("Robustness: Success Rate", fontsize=14)
    plt.xlim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot3_robustness.png"); plt.close()

def run_trace_for_plot(A, b, method_name, **kwargs):
    history = []
    def cb(xk):
        res = np.linalg.norm(A @ xk - b) / np.linalg.norm(b)
        history.append(res)
    N = A.shape[0]
    
    try:
        if method_name == 'CG':
            cg(A, b, callback=cb, rtol=1e-6, maxiter=2000)
        elif method_name == 'Jacobi':
            d = A.diagonal(); d[d==0]=1
            M = sp.diags(1/d)
            cg(A, b, M=M, callback=cb, rtol=1e-6, maxiter=2000)
        elif 'ILU' in method_name:
            tol = kwargs.get('tol', 1e-4)
            #CSC for ILU setup, you like it Misha, yes?
            ilu = spilu(A.tocsc(), drop_tol=tol, fill_factor=20)
            M = LinearOperator((N,N), matvec=ilu.solve)
            cg(A, b, M=M, callback=cb, rtol=1e-6, maxiter=2000)
        elif 'RIC2S' in method_name:
            tau = kwargs.get('tau', 0.1)
            _, hist, _ = RIC2CG.solve(A, b, stable=True, tau=tau, rtol=1e-6, maxiter=2000)
            history = hist
    except Exception as e:
        print(f"    ! {method_name} failed: {e}")
        return None
    
    return history

def plot_convergence_history(original_df):
    print("Generating Plot 2: Convergence History (Multi-matrix)...")
    
    available_df = original_df[original_df['path'].apply(os.path.exists)].sort_values('rows', ascending=False)
    
    if len(available_df) < 2:
        print("Not enough matrices for multiple plots. Plotting what we have.")
        targets = available_df
    else:
        
        targets = pd.concat([available_df.iloc[[0]], available_df.iloc[[len(available_df)//2]]])

    for i, (_, row) in enumerate(targets.iterrows()):
        name = row['name']
        path = row['path']
        print(f"  > Processing Matrix {i+1}: {name} (Rows: {row['rows']})")
        
        try:
            A = mmread(path).tocsr()
            b = A @ np.random.random(A.shape[0])#np.ones(A.shape[0])
            traces = {}
            
            traces['CG'] = run_trace_for_plot(A, b, 'CG')
            traces['Jacobi'] = run_trace_for_plot(A, b, 'Jacobi')
            
            traces['ILU (1e-3)'] = run_trace_for_plot(A, b, 'ILU', tol=1e-3)
            traces['ILU (1e-5)'] = run_trace_for_plot(A, b, 'ILU', tol=1e-5)
            
            traces['RIC2S (tau=0.05)'] = run_trace_for_plot(A, b, 'RIC2S', tau=0.05)
            
            plt.figure(figsize=(12, 7))
            for label, history in traces.items():
                if history is None or len(history) == 0: continue
                plt.plot(history, label=label, linewidth=2, alpha=0.8)
                
            plt.yscale('log')
            plt.title(f"Convergence History: {name}", fontsize=14)
            plt.xlabel("Iterations", fontsize=12)
            plt.ylabel("Relative Residual (Log10)", fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            filename = f"{OUTPUT_DIR}/plot2_history_{name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"    Saved {filename}")
            
        except Exception as e:
            print(f"    Error processing {name}: {e}")
	
def main():
    print("Loading Results...")
    long_df, original_df = load_and_preprocess()
    
    plot_pareto_frontier(long_df)
    plot_robustness(long_df)
    plot_mean_performance_by_size(long_df)
    plot_convergence_history(original_df)
    
    print(f"\nAll plots saved to '{OUTPUT_DIR}/'")
    print(r"""
 ____  _____ _____ ______   _   _ _   _ _____ ____  
|  _ \| ____| ____|___  /  | \ | | | | |_   _/ ___| 
| | | |  _| |  _|    / /   |  \| | | | | | | \___ \ 
| |_| | |___| |___  / /_   | |\  | |_| | | |  ___) |
|____/|_____|_____|/____|  |_| \_|\___/  |_| |____/ 
""")
if __name__ == "__main__":
    main()
