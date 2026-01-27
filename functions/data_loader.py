import os
import pandas as pd
import numpy as np

def load_cancer_data(relative_path='smiles_data.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, relative_path)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found at {csv_path}. "
            "Please place your SMILES CSV file in the root directory "
            "or specify the path using --fn_path."
        )
        
    df = pd.read_csv(csv_path)
    
    if 'SMILES' not in df.columns or 'DockingScore' not in df.columns:
        raise ValueError("CSV file is missing 'SMILES' or 'DockingScore' columns")

    original_len = len(df)
    
    nan_rows = df[df['SMILES'].isna()]
    
    if not nan_rows.empty:
        print("-" * 60)
        print(f"!!! WARNING: Found {len(nan_rows)} rows with empty SMILES (NaN) !!!")
        print(f"    Empty row indices: {nan_rows.index.tolist()}")
        print("    Removing these rows...")
        print("-" * 60)

    df = df.dropna(subset=['SMILES']) 
    
    df['SMILES'] = df['SMILES'].astype(str)
    
    nan_dropped = original_len - len(df)

    mask_outliers = df['DockingScore'] > 100
    outlier_count = mask_outliers.sum()
    
    if outlier_count > 0:
        df.loc[mask_outliers, 'DockingScore'] = 0.0
    
    print(f"[Data Loader] Loaded file: {relative_path}")
    print(f"              Original rows: {original_len}")
    
    if nan_dropped > 0:
        print(f"              !!! Removed empty SMILES: {nan_dropped} rows (fixed float error)")
        
    if outlier_count > 0:
        print(f"              !!! Corrected outliers: {outlier_count} counts (Score > 100 -> Set to 0.0)")
        
    print(f"              Effective rows: {len(df)}")
    
    return df['SMILES'].tolist(), df['DockingScore'].to_numpy()
