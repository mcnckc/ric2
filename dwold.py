import ssgetpy
import os
import pandas as pd
import numpy as np
from scipy.io import mmread
from nlass import NLASS 
import matplotlib.pyplot as plt
import seaborn as sns
import sys 

download_folder = 'matrices_collection'
registry_file = 'matrix_registry.csv'

os.makedirs(download_folder, exist_ok=True)

print("Searching SuiteSparse collection...")
results = ssgetpy.search(
    rowbounds=(1000, 50000), 
    isspd=True, # SPD + real
    limit=10,  # Change to None to download ALL (approx 200 matrices)
    kind='Structural Problem' # PHYSUCS RULEES
)
print(f"Found {len(results)} matrices.")
user_input = input("Start the loading process? (y/n): ")

if user_input.lower() == 'n':
    sys.exit()
else:
    pass

print(f"Downloading to '{download_folder}'... (This WILL take time)")
results.download(destpath=download_folder, extract=True)
print("Download complete.")

print("Generating registry CSV...")
data = []

for matrix in results:
    
    mtx_path = os.path.join(download_folder, matrix.name, f"{matrix.name}.mtx")
    
    data.append({
        'name': matrix.name,
        'group': matrix.group,
        'rows': matrix.rows,
        'cols': matrix.cols,
        'nnz': matrix.nnz,
        'cond_est': 'Unknown', # Placeholder for later benchmarking
        'path': mtx_path
    })

df = pd.read_csv(registry_file) if os.path.exists(registry_file) else pd.DataFrame(data)
# Create registryssssss
if not os.path.exists(registry_file):
    df = pd.DataFrame(data)
    df.to_csv(registry_file, index=False)
    print(f"Created new registry: {registry_file}")
else:
    pd.DataFrame(data).to_csv(registry_file, index=False)
    print(f"Updated registry: {registry_file}")


print("\nPreview of dataset:")
print(df[['name', 'rows', 'nnz', 'status']].head())