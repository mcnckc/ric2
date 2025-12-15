import pandas as pd
import os
import numpy as np
from scipy.io import mmread
from nlass import NLASS
import scipy.sparse as sp
REGISTRY_FILE = 'matrix_registry.csv'
RESULTS_FILE = 'benchmark_results.csv'

def init_results_file():
    # if os.path.exists(RESULTS_FILE):
    #     return pd.read_csv(RESULTS_FILE)
    if not os.path.exists(REGISTRY_FILE):
        raise FileNotFoundError("Registry not found. Run the downloader first.")
    df = pd.read_csv(REGISTRY_FILE)[['name', 'group', 'rows', 'cols', 'nnz', 'path']]
    df.to_csv(RESULTS_FILE, index=False)
    return df

def _load_and_update(run_logic_name, task_function):
    df = pd.read_csv(RESULTS_FILE)
    print(f"--- {run_logic_name} ---")
    for i, row in df.iterrows():
        if not os.path.exists(row['path']):
            continue
        print(f"[{i+1}/{len(df)}] {row['name']}...", end=" ", flush=True)
        try:
            A = mmread(row['path'])
            solver = NLASS(A)
            new_results = task_function(solver, row)
            
            for col, val in new_results.items():
                if col not in df.columns:
                    df[col] = np.nan
                df.at[i, col] = val
            
            df.to_csv(RESULTS_FILE, index=False)
            print("Saved")
        except Exception as e:
            print(f"Failed: {e}")

def run_baselines():
    def logic(solver, row):
        res = {}
        res.update(solver.test_cg_none())
        res.update(solver.test_cg_jacobi())
        res.update(solver.test_cg_sgs())
        res.update(solver.test_exact())
        return res
    _load_and_update("Baselines", logic)

def run_ilu_sensitivity():
    def logic(solver, row):
        res = {}
        for tol in [1e-3, 1e-4, 1e-5, 1e-6]:
            res.update(solver.test_cg_ilu(drop_tol=tol))
        return res
    _load_and_update("ILU_Sensitivity", logic)

def run_ric2s_sensitivity():
    def logic(solver, row):
        res = {}
        for t in [0.05, 0.01, 0.001]:
            res.update(solver.test_ric2(stable=True, tau=t))
        return res
    _load_and_update("RIC2S_Sensitivity", logic)

def run_exact_cholesky_ric():
    def logic(solver, row):
        if row['rows'] > 2000:
            return {}
        return solver.test_ric2(stable=True, tau=0.0)
    _load_and_update("Exact_Cholesky_RIC", logic)