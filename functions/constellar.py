import os
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONSTEL_CACHE_DIR = os.path.join(_THIS_DIR, "cache")
os.makedirs(_CONSTEL_CACHE_DIR, exist_ok=True)

def load_constellar_data_discrete(target_col: str = "metrics.qi", max_samples=None):
    safe_target = target_col.replace(".", "_")
    tag = "all" if max_samples is None else f"n{max_samples}"

    cache_name = f"constel_data_{safe_target}_{tag}.npz"
    cache_path = os.path.join(_CONSTEL_CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        print(f"[constellar] Loading cached data from {cache_path}")
        data = np.load(cache_path)
        X = data["X"]
        y = data["y"]
        print(f"[constellar] Range Y (cached): [{y.min():.4f}, {y.max():.4f}]")
        return X, y

    from datasets import load_dataset
    print("[constellar] Downloading/Loading ConStellaration dataset...")
    ds = load_dataset("proxima-fusion/constellaration", "default", split="train")

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    df = ds.to_pandas()

    mask = ~pd.isna(df[target_col])
    df = df.loc[mask]

    print("[constellar] Processing features (Flattening R/Z coefficients)...")

    r_list, z_list, y_list = [], [], []
    for _, row in df.iterrows():
        try:
            r_val = np.hstack(row["boundary.r_cos"]).astype(np.float32)
            z_val = np.hstack(row["boundary.z_sin"]).astype(np.float32)
            if np.isnan(r_val).any() or np.isnan(z_val).any():
                continue
            r_list.append(r_val)
            z_list.append(z_val)
            y_list.append(row[target_col])
        except Exception:
            continue

    if len(y_list) == 0:
        raise RuntimeError("[constellar] No valid samples found after filtering NaNs / parsing rows.")

    max_r = max(len(v) for v in r_list)
    max_z = max(len(v) for v in z_list)

    n_samples = len(y_list)

    X_r = np.zeros((n_samples, max_r), dtype=np.float32)
    X_z = np.zeros((n_samples, max_z), dtype=np.float32)

    valid_r = np.zeros((n_samples, max_r), dtype=bool)
    valid_z = np.zeros((n_samples, max_z), dtype=bool)

    for i in range(n_samples):
        lr = len(r_list[i])
        lz = len(z_list[i])

        X_r[i, :lr] = r_list[i]
        X_z[i, :lz] = z_list[i]

        valid_r[i, :lr] = True
        valid_z[i, :lz] = True

    X_full = np.hstack([X_r, X_z])
    valid = np.hstack([valid_r, valid_z])

    keep = valid.any(axis=0)
    X = X_full[:, keep]

    y = np.array(y_list, dtype=np.float32)
    print(f"[constellar] Range Y: [{y.min():.4f}, {y.max():.4f}]")

    lb = X.min(axis=0)
    ub = X.max(axis=0)
    width = ub - lb
    eps0 = 1e-8

    const_idx = np.where(width <= eps0)[0]
    print("[constellar] constant dim indices =", const_idx.tolist())

    if const_idx.size > 0:
        vals = X[:, const_idx]
        print("[constellar] constant dim values =", vals[0].tolist())
        print("[constellar] constant dim (lb,ub) =",
              list(zip(lb[const_idx].tolist(), ub[const_idx].tolist())))

    keep2 = width > eps0
    X = X[:, keep2]
    print("[constellar] Dropped constant dims:", (~keep2).sum(), "=> New X shape:", X.shape)

    lb = X.min(axis=0)
    ub = X.max(axis=0)
    width = ub - lb
    print(f"[constellar] Zero-width dims (after drop): {(width <= eps0).sum()} | "
          f"min width={width.min():.6g}  max width={width.max():.6g}")

    np.savez_compressed(cache_path, X=X, y=y)

    print(f"[constellar] Data cached to {cache_path}")
    return X, y
