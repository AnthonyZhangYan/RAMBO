import math
import torch
import torch.nn as nn
import torch.optim as optim

class RBFKernel:
    def __init__(self, lengthscale=1.0, variance=1.0, device="cpu"):
        self.device = torch.device(device)
        ls = torch.as_tensor(lengthscale, dtype=torch.float32, device=self.device)
        if ls.ndim == 0:
            self.lengthscale = ls.view(1)
        else:
            self.lengthscale = ls.clone()
        self.variance = torch.as_tensor(float(variance), dtype=torch.float32, device=self.device)

    def _scale(self, X):
        ls = self.lengthscale
        return X / ls if ls.numel() > 1 else X / ls.item()

    def K(self, X, Z=None):
        if Z is None:
            Z = X
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        Xs, Zs = self._scale(X), self._scale(Z)
        x2 = (Xs * Xs).sum(dim=1, keepdim=True)
        z2 = (Zs * Zs).sum(dim=1, keepdim=True).T
        sq = x2 + z2 - 2.0 * (Xs @ Zs.T)
        return self.variance * torch.exp(-0.5 * sq)

    def K_diag(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self.variance.expand(X.shape[0])

class GPRegressor:
    def __init__(self, ard=True, normalize_y=True, jitter=1e-8, device="cpu"):
        self.ard = ard
        self.normalize_y = normalize_y
        self.jitter = jitter
        self.device = torch.device(device)
        self.theta_opt = None

        self._log_ls   = None
        self._log_var  = nn.Parameter(torch.tensor(math.log(1.0), dtype=torch.float32, device=self.device))
        self._log_noise= nn.Parameter(torch.tensor(math.log(1e-2), dtype=torch.float32, device=self.device))

        # caches
        self.Xn = None
        self.yc = None
        self.x_mean = None
        self.x_std  = None
        self.y_mean = None
        self.y_std  = None
        self.kernel = None
        self.L = None
        self.alpha = None

    def _set_dims_and_kernel(self, X, init_lengthscale, init_variance):
        N, D = X.shape
        if self.ard:
            ls0 = torch.full((D,), float(init_lengthscale), device=self.device)
        else:
            ls0 = torch.tensor([float(init_lengthscale)], device=self.device)
        self._log_ls = nn.Parameter(ls0.log())
        self._log_var.data = torch.tensor(math.log(float(init_variance)), device=self.device)
        self.kernel = RBFKernel(torch.exp(self._log_ls).detach(),
                                torch.exp(self._log_var).detach(),
                                device=self.device)
        self.N, self.D = N, D

    @property
    def noise(self):
        return torch.exp(self._log_noise)

    def _update_kernel_params(self):
        with torch.no_grad():
            self.kernel.lengthscale = torch.exp(self._log_ls)
            self.kernel.variance    = torch.exp(self._log_var)

    def _nll(self, Xn, yc):
        self._update_kernel_params()
        N = Xn.size(0)
        K = self.kernel.K(Xn)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
        K = 0.5 * (K + K.T)
        K = K + (self.noise + self.jitter) * torch.eye(N, device=self.device)

        jitter = 0.0
        for _ in range(8):
            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(N, device=self.device))
                break
            except RuntimeError:
                jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
        else:
            L = torch.linalg.cholesky(K + 1e-2 * torch.eye(N, device=self.device))
        alpha = torch.cholesky_solve(yc.unsqueeze(-1), L).squeeze(-1)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        nll = 0.5 * (yc @ alpha) + 0.5 * logdet + 0.5 * N * math.log(2.0 * math.pi)
        return nll, L, alpha

    def _constrain_logs(self):
        lo_ls, hi_ls = math.log(1e-3), math.log(1e3)
        lo_v,  hi_v  = math.log(1e-6), math.log(1e6)
        lo_n,  hi_n  = math.log(1e-8), math.log(1e1)
        with torch.no_grad():
            for p, lo, hi, back in [
                (self._log_ls, lo_ls, hi_ls, 0.0),
                (self._log_var, lo_v,  hi_v, 0.0),
                (self._log_noise, lo_n, hi_n, math.log(1e-2)),
            ]:
                p.data = torch.nan_to_num(p.data, nan=back, posinf=hi, neginf=lo)
                p.data.clamp_(lo, hi)

    def fit(self, X, y, init_lengthscale=1.0, init_variance=1.0, init_noise=1e-2, bounds=None, steps=300, lr=0.05, verbose=False):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).view(-1)
        self._set_dims_and_kernel(X, init_lengthscale, init_variance)
        self._log_noise.data = torch.tensor(math.log(float(init_noise)), device=self.device)

        if self.normalize_y:
            self.y_mean = y.mean()
            self.y_std  = y.std().clamp_min(1e-12)
        else:
            self.y_mean = torch.tensor(0.0, device=self.device)
            self.y_std  = torch.tensor(1.0, device=self.device)
        yc = (y - self.y_mean) / self.y_std

        self.x_mean = X.mean(dim=0, keepdim=True)
        self.x_std  = X.std(dim=0, keepdim=True).clamp_min(1e-12)
        Xn = (X - self.x_mean) / self.x_std

        opt = optim.LBFGS(
            [self._log_ls, self._log_var, self._log_noise],
            max_iter=steps,
            history_size=20,
            line_search_fn="strong_wolfe"
        )

        def closure():
            self._constrain_logs()
            opt.zero_grad()
            nll, L_, alpha_ = self._nll(Xn, yc)
            self.L, self.alpha = L_, alpha_
            nll.backward()
            return nll

        opt.step(closure)

        with torch.no_grad():
            self._constrain_logs()
            nll, L, alpha = self._nll(Xn, yc)
        self.L = L
        self.alpha = alpha

        self.Xn = Xn
        self.yc = yc
        self.theta_opt = torch.cat([
            self._log_ls.detach(),
            self._log_var.detach().view(1),
            self._log_noise.detach().view(1)
        ])
        return self

    def log_marginal_likelihood(self):
        nll, _, _ = self._nll(self.Xn, self.yc)
        return -nll

    def predict(self, Xstar, return_std=False, return_cov=False, include_noise=False):
        Xs = torch.as_tensor(Xstar, dtype=torch.float32, device=self.device)
        Xsn = (Xs - self.x_mean) / self.x_std
        self._constrain_logs()
        self._update_kernel_params()

        # normalized space
        Kxs = self.kernel.K(Xsn, self.Xn)
        Kxs = torch.nan_to_num(Kxs, nan=0.0, posinf=1e6, neginf=-1e6)
        mu_n = Kxs @ self.alpha

        if not (return_std or return_cov):
            mu = mu_n * self.y_std + self.y_mean
            return mu

        v = torch.cholesky_solve(Kxs.T, self.L)
        Kss_diag = self.kernel.K_diag(Xsn)
        var_n = Kss_diag - (Kxs * v.T).sum(dim=1)
        var_n = torch.nan_to_num(var_n, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        if return_cov:
            Kss_n = self.kernel.K(Xsn)
            cov_n = Kss_n - (Kxs @ v)
            cov_n = 0.5 * (cov_n + cov_n.T)
            cov_n.diagonal().clamp_min_(0.0)
            cov = cov_n * (self.y_std ** 2)
            if include_noise:
                Nstar = Xs.shape[0]
                cov = cov + (self.noise * (self.y_std ** 2)) * torch.eye(Nstar, device=self.device)
            mu = mu_n * self.y_std + self.y_mean
            return mu, cov

        if include_noise:
            var_n = var_n + self.noise
        std = torch.sqrt(var_n.clamp_min(0.0)) * self.y_std
        mu  = mu_n * self.y_std + self.y_mean
        return mu, std
