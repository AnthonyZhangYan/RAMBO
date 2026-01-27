import math
import torch
from models.sgp import GPRegressor
from models.rambo import DPMGPR

def ei(X, gp: GPRegressor, y_best, xi=0.1, maximize=True):
    X = torch.as_tensor(X, dtype=torch.float32, device=gp.device)
    mu, std = gp.predict(X, return_std=True, include_noise=False)
    
    mu  = torch.nan_to_num(mu,  nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.nan_to_num(std, nan=1e-6, posinf=1e6, neginf=1e-6)
    std = std.clamp_min(1e-12)
    
    s = 1.0 if maximize else -1.0
    yb = y_best if torch.is_tensor(y_best) else torch.tensor(float(y_best), device=mu.device)
    imp = s * (mu - yb) - xi
    Z = imp / std
    N01 = torch.distributions.Normal(0.0, 1.0)
    return imp * N01.cdf(Z) + std * torch.exp(N01.log_prob(Z))

def ei_mix(model: DPMGPR, X, f_best, xi=0.1):
    mu, std, w_k, mu_k, std_k = model.predict_torch(
        X, return_std=True, return_components=True
    )

    device, dtype = mu_k.device, mu_k.dtype
    f_best_t = torch.as_tensor(f_best, device=device, dtype=dtype)
    xi_t     = torch.as_tensor(xi,      device=device, dtype=dtype)

    w_k = torch.nan_to_num(w_k, nan=0.0, posinf=0.0, neginf=0.0)
    w_sum = w_k.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w_k = w_k / w_sum

    mu_k  = torch.nan_to_num(mu_k,  nan=0.0, posinf=0.0, neginf=0.0)
    std_k = torch.nan_to_num(std_k, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-12)

    gamma = (mu_k - f_best_t - xi_t) / std_k
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

    normal = torch.distributions.Normal(
        torch.zeros((), device=device, dtype=dtype),
        torch.ones ((), device=device, dtype=dtype)
    )
    Phi = normal.cdf(gamma)
    phi = torch.exp(normal.log_prob(gamma))

    ei_k = std_k * (gamma * Phi + phi)
    return (w_k * ei_k).sum(dim=1)

def optimize_acq_lbfgs(acq_of_X, bounds, n_starts=20, max_iter=60, device="cpu"):
    bounds = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = (ub - lb)
    D = bounds.shape[0]

    best_x = None
    best_val = -1e30

    sobol = torch.quasirandom.SobolEngine(dimension=D, scramble=True)
    U = sobol.draw(n_starts).to(device=device)
    Xstarts = lb + span * U

    for i in range(n_starts):
        x0 = Xstarts[i:i+1]
        eps = 1e-6
        u0 = ((x0 - lb) / span).clamp(eps, 1 - eps)
        z0 = (u0 / (1 - u0)).log()
        z = z0.detach().clone().requires_grad_(True)

        adam = torch.optim.Adam([z], lr=0.05)
        for _ in range(25):
            adam.zero_grad()
            x = lb + span * torch.sigmoid(z)
            val = acq_of_X(x).reshape(())
            (-val).backward()
            adam.step()

        opt = torch.optim.LBFGS([z], max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            x = lb + span * torch.sigmoid(z)
            val = acq_of_X(x).reshape(())
            (-val).backward()
            return -val
        opt.step(closure)

        with torch.no_grad():
            x = lb + span * torch.sigmoid(z)
            v = acq_of_X(x).item()
            if v > best_val:
                best_val, best_x = v, x.detach().squeeze(0)

    return best_x

def optimize_acq_pool(
    acq_of_X,
    pool_X,
    *,
    X_seen=None,
    avoid_duplicates=True,
    tol=1e-12,
    device="cpu",
    batch_size=None
):
    pool = torch.as_tensor(pool_X, dtype=torch.float32, device=device)
    N = pool.shape[0]

    mask = torch.ones(N, dtype=torch.bool, device=device)

    if avoid_duplicates and (X_seen is not None):
        Xs = torch.as_tensor(X_seen, dtype=torch.float32, device=device)
        if Xs.numel() > 0:
            chunk = 4096
            dup = torch.zeros(N, dtype=torch.bool, device=device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                d = torch.cdist(pool[s:e], Xs)
                dup[s:e] = (d.min(dim=1).values <= tol)
            mask = ~dup
            
    idx_all = torch.arange(N, device=device)[mask]
    if idx_all.numel() == 0:
        raise ValueError("No candidates left in pool after removing duplicates.")

    best_val = -1e30
    best_x = None
    best_idx = None

    if batch_size is None:
        cands = pool[idx_all]
        vals = acq_of_X(cands).view(-1)
        j = torch.argmax(vals)
        best_val = float(vals[j])
        best_x = cands[j].detach()
        best_idx = int(idx_all[j])
    else:
        for s in range(0, idx_all.numel(), batch_size):
            ids = idx_all[s:s + batch_size]
            cands = pool[ids]
            vals = acq_of_X(cands).view(-1)
            j = torch.argmax(vals)
            v = float(vals[j])
            if v > best_val:
                best_val = v
                best_x = cands[j].detach()
                best_idx = int(ids[j])

    return best_x, best_val, best_idx

def bayes_opt(
    f_obj,
    bounds,
    *,
    mode="sgp",
    model=None,
    n_init=8,
    n_iter=32,
    acq="ei",
    xi=0.1,
    maximize=True,
    gp_kwargs=None,
    dpmm_fit_kwargs=None,
    X0=None, y0=None,
    device="cpu",
    rng=None,
    alpha_schedule=None,
    pool_X=None,
    pool_Y=None,
    avoid_duplicates=True,
    pool_tol=1e-6,
    pool_batch_size=None
):
    device = torch.device(device)
    if gp_kwargs is None: gp_kwargs = {}
    if dpmm_fit_kwargs is None: dpmm_fit_kwargs = {}
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(42)

    mode = str(mode).lower()
    is_dpmm = (mode == "dpmm")

    if is_dpmm:
        if model is None:
            raise ValueError("mode='dpmm' requires a DPMGPR instance passed to model.")
        if not hasattr(model, "predict_torch"):
            raise ValueError("DPMM model must implement predict_torch(...) to support gradients for X.")
    else:
        model = GPRegressor(ard=True, normalize_y=True, device=device)

    bounds = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    D = bounds.shape[0]

    pool_X_t = None
    pool_Y_t = None
    if pool_X is not None:
        pool_X_t = torch.as_tensor(pool_X, dtype=torch.float32, device=device)
        if pool_Y is not None:
            pool_Y_t = torch.as_tensor(pool_Y, dtype=torch.float32, device=device).view(-1)

    if X0 is not None and y0 is not None:
        X = torch.as_tensor(X0, dtype=torch.float32, device=device)
        y = torch.as_tensor(y0, dtype=torch.float32, device=device).view(-1)
    else:
        if pool_X_t is not None:
            Np = pool_X_t.shape[0]
            perm = torch.randperm(Np, generator=rng, device=device)
            idx0 = perm[:n_init]
            X = pool_X_t[idx0]
            if pool_Y_t is not None:
                y = pool_Y_t[idx0]
            else:
                y = f_obj(X).view(-1)
        else:
            X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(n_init, D, generator=rng, device=device)
            y = f_obj(X).view(-1)

    if is_dpmm:
        iters_fit   = dpmm_fit_kwargs.pop("n_iterations", 150)
        burn_in     = dpmm_fit_kwargs.pop("burn_in", 75)
        optim_every = dpmm_fit_kwargs.pop("optim_every", 5)
        model.fit(X, y, n_iterations=iters_fit, burn_in=burn_in, optim_every=optim_every, **dpmm_fit_kwargs)
    else:
        model.fit(X, y, **gp_kwargs)

    y_best = torch.max(y) if maximize else torch.min(y)
    hist = [{"iter": 0, "x": X.clone(), "y": y.clone(), "best": float(y_best)}]
    print(f"[INIT] best={float(y_best):.6f}")

    for t in range(1, n_iter + 1):
        print(f"\n[BO iteration {t}/{n_iter}] fitting {'DPMM-GP' if is_dpmm else 'Single GP'} on n={X.shape[0]} points =>\n")
        
        if is_dpmm and (alpha_schedule is not None):
            model.alpha = float(alpha_schedule(t))
            print(f"[DPMM] alpha(t={t}) = {model.alpha:.6f}")
        if is_dpmm:
            model.fit(X, y, n_iterations=iters_fit, burn_in=burn_in, optim_every=optim_every, **dpmm_fit_kwargs)
        else:
            model.fit(X, y, **gp_kwargs)

        y_best = torch.max(y) if maximize else torch.min(y)

        if is_dpmm:
            if acq.lower() == "ei":
                acq_fn_core = lambda Xc: ei_mix(model, Xc, y_best, xi=xi)
            else:
                raise ValueError("acq must be 'ei' in dpmm mode (UCB removed)")
        else:
            if acq.lower() == "ei":
                acq_fn_core = lambda Xc: ei(Xc, model, y_best, xi=xi, maximize=maximize)
            else:
                raise ValueError("acq must be 'ei' in sgp mode (UCB removed)")

        if pool_X_t is not None:
            if pool_batch_size is None:
                pool_batch_size = 2048
            x_next, _, pool_idx = optimize_acq_pool(
                acq_fn_core,
                pool_X_t,
                X_seen=X,
                avoid_duplicates=avoid_duplicates,
                tol=pool_tol,
                device=device,
                batch_size=pool_batch_size
            )
            if pool_Y_t is not None:
                y_next = pool_Y_t[pool_idx].reshape(1)
            else:
                y_next = f_obj(x_next.unsqueeze(0)).reshape(1)
        else:
            x_next = optimize_acq_lbfgs(acq_fn_core, bounds, n_starts=20, max_iter=60, device=device)
            y_next = f_obj(x_next.unsqueeze(0)).reshape(1)

        X = torch.cat([X, x_next.unsqueeze(0)], dim=0)
        y = torch.cat([y, y_next], dim=0)

        y_best = torch.max(y) if maximize else torch.min(y)
        hist.append({"iter": t, "x": X.clone(), "y": y.clone(), "best": float(y_best)})
        print(f"[BO] iter {t:3d}/{n_iter}  best={float(y_best):.6f}")

    i_best = torch.argmax(y) if maximize else torch.argmin(y)
    return X[i_best], y[i_best], X, y, hist
