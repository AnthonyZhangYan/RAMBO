import numpy as np
import torch
import os

def levy_torch(X):
    X = torch.as_tensor(X, dtype=torch.float32)
    w = 1 + (X - 1.0) / 4.0
    term1 = torch.sin(torch.pi * w[..., 0])**2
    term_mid = torch.sum(
        (w[..., :-1] - 1.0)**2 * (1 + 10 * torch.sin(torch.pi * w[..., :-1] + 1.0)**2),
        dim=-1,
    )
    term_last = (w[..., -1] - 1.0)**2 * (1 + torch.sin(2 * torch.pi * w[..., -1])**2)
    return term1 + term_mid + term_last

def schwefel_torch(X):
    X = torch.as_tensor(X, dtype=torch.float32)
    d = X.shape[-1]
    return 418.9829 * d - torch.sum(X * torch.sin(torch.sqrt(torch.abs(X))), dim=-1)

def torsion_energy_torch(X, n=15):
    from functions.torsion_energy import torsion_energy as _scalar
    X = torch.as_tensor(X, dtype=torch.float32)
    orig_shape = X.shape[:-1]
    X_flat = X.reshape(-1, X.shape[-1])
    PENALTY_VALUE = -1e3
    ys = []
    for row in X_flat:
        try:
            y = _scalar(row, n=n)
            y_t = torch.as_tensor(y, dtype=torch.float32)
            if not torch.isfinite(y_t):
                y_t = torch.tensor(PENALTY_VALUE)
            else:
                y_t = y_t.clamp(min=PENALTY_VALUE, max=100.0)
        except Exception:
            y_t = torch.tensor(PENALTY_VALUE)
        ys.append(y_t)
    return torch.stack(ys).view(orig_shape)

_cancer_cache = None
def get_cancer_data(pca_dim=50):
    from functions.data_loader import load_cancer_data
    from functions.encoders import SMILESEncoder
    global _cancer_cache
    if _cancer_cache is not None: return _cancer_cache

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cancer_data_pca{pca_dim}.pt")

    if os.path.exists(cache_path):
        cache_data = torch.load(cache_path)
        _cancer_cache = (cache_data["X"], cache_data["Y"], cache_data["bounds"])
        return _cancer_cache

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    smiles, raw_scores = load_cancer_data()
    Y = torch.tensor(raw_scores, dtype=torch.float32).unsqueeze(-1) * -1.0
    encoder = SMILESEncoder(n_bits=2048, use_chirality=True, pca_dim=pca_dim)
    X = encoder.fit_transform(smiles).float()
    bounds = torch.tensor(np.stack([X.min(0)[0].cpu().numpy(), X.max(0)[0].cpu().numpy()]).T, dtype=torch.float32)
    torch.save({"X": X, "Y": Y, "bounds": bounds}, cache_path)
    _cancer_cache = (X, Y, bounds)
    return _cancer_cache

def cancer_6t2w_torch(X):
    X = torch.as_tensor(X, dtype=torch.float32)
    X_data, Y_data, _ = get_cancer_data()
    X_data, Y_data = X_data.to(X.device), Y_data.to(X.device)
    orig_shape = X.shape[:-1]
    X_flat = X.view(-1, X.shape[-1])
    diff = X_flat.unsqueeze(1) - X_data.unsqueeze(0)
    nn_idx = (diff ** 2).sum(-1).argmin(dim=1)
    return Y_data[nn_idx, 0].view(orig_shape)

_constellar_cache = {}
def get_constellar_data(target_col="metrics.qi"):
    global _constellar_cache
    if target_col in _constellar_cache: return _constellar_cache[target_col]
    from functions.constellar import load_constellar_data_discrete
    X_np, y_np = load_constellar_data_discrete(target_col=target_col)
    X, Y = torch.from_numpy(X_np).float(), torch.from_numpy(y_np).float().unsqueeze(-1)
    eps = 1e-4
    bounds = torch.tensor(np.stack([X.min(0)[0].numpy() - eps, X.max(0)[0].numpy() + eps]).T, dtype=torch.float32)
    _constellar_cache[target_col] = (X, Y, bounds)
    return _constellar_cache[target_col]

def constellaration_torch(X):
    X = torch.as_tensor(X, dtype=torch.float32)
    X_data, Y_data, _ = get_constellar_data()
    X_data, Y_data = X_data.to(X.device), Y_data.to(X.device)
    orig_shape = X.shape[:-1]
    X_flat = X.view(-1, X.shape[-1])
    nn_idx = torch.cdist(X_flat, X_data).argmin(dim=1)
    return Y_data[nn_idx, 0].view(orig_shape)

def _bounds_cancer():
    _, _, b = get_cancer_data()
    return b.detach().cpu().numpy().astype(np.float32)

def _bounds_constellaration():
    _, _, b = get_constellar_data(target_col="metrics.qi")
    return b.detach().cpu().numpy().astype(np.float32)

BOUNDS = {
    "levy6d": np.tile([[-10.0, 10.0]], (6, 1)).astype(np.float32),
    "levy10d": np.tile([[-10.0, 10.0]], (10, 1)).astype(np.float32),
    "schwefel6d": np.tile([[-500.0, 500.0]], (6, 1)).astype(np.float32),
    "schwefel10d": np.tile([[-500.0, 500.0]], (10, 1)).astype(np.float32),
    "torsion_energy": np.tile([[-180.0, 180.0]], (12, 1)).astype(np.float32),
    "cancer_6t2w": _bounds_cancer,
    "constellaration": _bounds_constellaration,
}

def make_f_obj(fn: str, negate: bool = True):
    fn = fn.lower()
    ftab = {
        "levy6d": levy_torch,
        "levy10d": levy_torch,
        "schwefel6d": schwefel_torch,
        "schwefel10d": schwefel_torch,
        "torsion_energy": torsion_energy_torch,
        "cancer_6t2w": cancer_6t2w_torch,
        "constellaration": constellaration_torch,
    }
    f = ftab[fn]
    if fn in ("torsion_energy", "cancer_6t2w", "constellaration"):
        return f
    return (lambda X: -f(X)) if negate else f

def get_bounds(fn: str, as_torch: bool = False, device: str = "cpu"):
    b = BOUNDS[fn.lower()]
    if callable(b): b = b()
    if as_torch: return torch.tensor(b, dtype=torch.float32, device=device)
    return b
