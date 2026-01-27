import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import shutil
import tempfile
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def _batch_compute_fp(smiles_list, n_bits, use_chirality):
    n = len(smiles_list)
    batch_fps = np.zeros((n, n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=n_bits, useChirality=use_chirality
            )
            batch_fps[i] = np.array(fp, dtype=np.float32)
    return batch_fps

class SMILESEncoder:
    def __init__(self, n_bits=2048, use_chirality=True, pca_dim=50, batch_size=4096):
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.pca_dim = pca_dim
        self.batch_size = batch_size
        self.ipca = None

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            self.n_jobs = int(slurm_cpus)
            print(f"[Encoder Config] Using {self.n_jobs} cores")
        else:
            try:
                self.n_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                self.n_jobs = max(1, cpu_count() - 1)
            print(f"[Encoder Config] Resource initialization complete. Using {self.n_jobs} cores")

    def _check_disk_space(self, path, required_gb=12):
        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (2**30)
            if free_gb < required_gb:
                print(f"[Warn] Directory {path} has low space ({free_gb}GB < {required_gb}GB).")
        except Exception:
            pass 

    def fit_transform(self, smiles_list):
        n_samples = len(smiles_list)
        print(f"[Encoder] Starting processing of {n_samples} ...")
        print(f"[Encoder] N_JOBS: {self.n_jobs}, Batch Size: {self.batch_size}")

        base_tmp = os.environ.get("TMPDIR", None)
        if base_tmp:
            self._check_disk_space(base_tmp, required_gb=12)
        
        temp_dir = tempfile.mkdtemp(dir=base_tmp)
        self._check_disk_space(temp_dir, required_gb=12)

        memmap_path = os.path.join(temp_dir, 'fps_cache.dat')
        
        try:
            X_memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n_samples, self.n_bits))
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to create temporary Memmap: {e}")

        try:
            chunk_size = self.batch_size
            n_chunks = int(np.ceil(n_samples / chunk_size))
            window_size = max(1, self.n_jobs * 2)
            
            print(f"[Encoder] Step 1/3: Generating fingerprints in parallel (Window Size={window_size})...")
            
            for w_start in tqdm(range(0, n_chunks, window_size), desc="Fingerprinting"):
                w_end = min(w_start + window_size, n_chunks)
                
                current_tasks = []
                task_indices = [] 
                
                for i in range(w_start, w_end):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, n_samples)
                    current_tasks.append(smiles_list[start:end])
                    task_indices.append((start, end))

                results = Parallel(
                    n_jobs=self.n_jobs, 
                    backend="loky", 
                    prefer="processes",
                    verbose=0
                )(
                    delayed(_batch_compute_fp)(batch_s, self.n_bits, self.use_chirality)
                    for batch_s in current_tasks
                )

                for (start, end), res_data in zip(task_indices, results):
                    X_memmap[start:end] = res_data
                
                del results, current_tasks
                
            X_memmap.flush()
            print("[Encoder] Fingerprint generation complete, saved to disk.")

            print("[Encoder] Step 2/3: Fitting PCA...")
            actual_dim = min(self.pca_dim, n_samples)
            self.ipca = IncrementalPCA(n_components=actual_dim, batch_size=self.batch_size)

            for i in tqdm(range(0, n_samples, self.batch_size), desc="PCA Fit"):
                batch_data = X_memmap[i : i + self.batch_size]
                self.ipca.partial_fit(batch_data)

            print("[Encoder] Step 3/3: Transforming data...")
            X_reduced = np.zeros((n_samples, actual_dim), dtype=np.float32)
            
            for i in tqdm(range(0, n_samples, self.batch_size), desc="PCA Transform"):
                batch_data = X_memmap[i : i + self.batch_size]
                res = self.ipca.transform(batch_data)
                n_res = res.shape[0]
                X_reduced[i : i + n_res] = res

        finally:
            print("[Encoder] Cleaning up temporary cache files...")
            try:
                if 'X_memmap' in locals():
                    del X_memmap
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Warn] Cleanup failed: {e}")

        return torch.tensor(X_reduced, dtype=torch.float32)
