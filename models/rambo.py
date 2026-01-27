import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import InverseGamma, Normal, Categorical
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def cholesky_solve(C, y):
    eps = 1e-9
    I = torch.eye(C.size(0), dtype=C.dtype, device=C.device)
    for tries in range(8):
        jitter = (eps * (10.0 ** tries)) if tries > 0 else 0.0
        try:
            L = torch.linalg.cholesky(C + jitter * I)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
            return alpha, L, logdet
        except RuntimeError:
            continue
    raise RuntimeError("Cholesky failed after jitter retries")

class GPKernel(nn.Module):
    def __init__(self, 
                 sigma_f=1.0, 
                 length_scale=1.0, 
                 sigma_n=0.1, 
                 device='cpu'
                 ):
        super().__init__()
        self.device = device
        
        self.log_sigma_f = nn.Parameter(torch.tensor(np.log(sigma_f), device=device))
        self.log_length_scale = nn.Parameter(torch.tensor(np.log(length_scale), device=device))
        self.log_sigma_n = nn.Parameter(torch.tensor(np.log(sigma_n), device=device))
    
    @property
    def sigma_f(self):
        return torch.exp(self.log_sigma_f)
    
    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)
    
    @property
    def sigma_n(self):
        return torch.exp(self.log_sigma_n)
    
    def forward(self, X1, X2, noise=False):
        X1_norm = (X1**2).sum(1).view(-1, 1)
        X2_norm = (X2**2).sum(1).view(1, -1) 
        sqdist = X1_norm + X2_norm - 2.0 * torch.mm(X1, X2.t())
        K = self.sigma_f**2 * torch.exp(-0.5 * sqdist / self.length_scale**2)  

        if noise:
            K = K + (self.sigma_n**2 + 1e-6) * torch.eye(K.shape[0], device=self.device) 
        return K  
    
    def get_params_dict(self):
        return {
            'sigma_f': self.sigma_f.item(),
            'length_scale': self.length_scale.item(),
            'sigma_n': self.sigma_n.item()
        }


class DPMGPR:
    def __init__(self, alpha=1.0, prior_params=None, device='cpu', use_grad_optim=True):
        self.alpha = alpha
        self.device = torch.device(device)
        self.use_grad_optim = use_grad_optim
        
        if prior_params is None:
            self.prior_params = {
                'sigma_f': (3.0, 2.0),  
                'length_scale': (3.0, 1.5), 
                'sigma_n': (3.0, 0.3)
            }
        else:
            self.prior_params = prior_params
            
        self.X = None
        self.y = None
        self.n = None
        self.z = None
        self.kernels = {}
        self.K = 0
        
        self.samples_z = []
        self.samples_K = []
        self.log_likelihoods = []
        
    def _initialize_kernel(self,):
        sigma_f = InverseGamma(
            self.prior_params['sigma_f'][0], 
            self.prior_params['sigma_f'][1]
        ).sample().item() 
        
        length_scale = InverseGamma(
            self.prior_params['length_scale'][0],
            self.prior_params['length_scale'][1]
        ).sample().item()  
        
        sigma_n = InverseGamma(
            self.prior_params['sigma_n'][0],
            self.prior_params['sigma_n'][1]
        ).sample().item()
        
        return GPKernel(sigma_f, length_scale, sigma_n, self.device)
        
    
    def _initialize_clusters(self, kmeans_iterations=15):
        X_np = self.X.cpu().numpy()
        n_init_clusters = min(3, self.n // 10)
        n_init_clusters = max(1, n_init_clusters)
    
        if n_init_clusters == 1:
            self.z = torch.zeros(self.n, dtype=torch.long, device=self.device) 
        else:
            from scipy.spatial.distance import cdist
            indices = np.random.choice(self.n, n_init_clusters, replace=False)
            centers = X_np[indices]
            
            for iteration in range(kmeans_iterations):
                distances = cdist(X_np, centers)
                z_np = np.argmin(distances, axis=1)

                for k in range(n_init_clusters):
                    mask = (z_np == k)
                    if np.sum(mask) > 0:
                        centers[k] = np.mean(X_np[mask], axis=0)
                
            self.z = torch.tensor(z_np, dtype=torch.long, device=self.device) 
        
        self.K = len(torch.unique(self.z))

        for k in range(self.K):
            self.kernels[k] = self._initialize_kernel()

    def _gp_predict(self, X_test, X_train, y_train, kernel):
        K         = kernel(X_train, X_train, noise=True)
        k_star    = kernel(X_train, X_test,  noise=False)
        k_star_ss = kernel(X_test,  X_test,  noise=False)

        try:
            alpha, L, _ = cholesky_solve(K, y_train)
            mu  = k_star.t() @ alpha
            v   = torch.linalg.solve_triangular(L, k_star, upper=False)
            var = torch.diag(k_star_ss) - (v**2).sum(dim=0)
            mu  = torch.nan_to_num(mu,  nan=0.0, posinf=0.0, neginf=0.0)
            var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)
            return mu, var
        except RuntimeError:
            diag = torch.diag(k_star_ss)
            var  = torch.nan_to_num(diag, nan=1.0, posinf=1e6, neginf=1.0).clamp_min(1e-6)
            mu   = diag * 0.0
            return mu, var
            
    def _sample_cluster_assignment(self, i):
        mask_without_i = torch.arange(self.n, device=self.device) != i
        z_without_i = self.z[mask_without_i] # (n-1,)
        
        log_probs = []
        clusters = []
        
        for k in range(self.K):
            n_k = torch.sum(z_without_i == k).item()
            if n_k > 0:
                log_prior = np.log(n_k / (self.n - 1 + self.alpha))
                mask_k = (self.z == k) & mask_without_i 
                X_k = self.X[mask_k]  # (n_k, d)
                y_k = self.y[mask_k]  # (n_k,)

                if len(X_k) > 0:
                    mu_pred, var_pred = self._gp_predict(self.X[i:i+1],  X_k, y_k, self.kernels[k])
                    log_lik = Normal(mu_pred[0], torch.sqrt(var_pred[0])).log_prob(self.y[i])
                    log_lik = log_lik.item()
                else:
                    sigma = self.kernels[k].sigma_f + self.kernels[k].sigma_n
                    log_lik = Normal(0.0, sigma).log_prob(self.y[i]).item()
                    
                log_probs.append(log_prior + log_lik)
                clusters.append(k)
        
        log_prior = np.log(self.alpha / (self.n - 1 + self.alpha))
        n_mc = 5
        log_liks = []
        kernel_samples = []
        for _ in range(n_mc):
            kernel_new = self._initialize_kernel()
            kernel_samples.append(kernel_new)
            sigma = kernel_new.sigma_f + kernel_new.sigma_n
            log_lik = Normal(0.0, sigma).log_prob(self.y[i]).item()
            log_liks.append(log_lik)
        
        log_liks = np.array(log_liks)
        log_lik_avg = np.max(log_liks) + np.log(np.mean(np.exp(log_liks - np.max(log_liks))))
        
        log_probs.append(log_prior + log_lik_avg)
        clusters.append(self.K)
        
        log_probs = np.array(log_probs)
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        
        new_cluster = np.random.choice(clusters, p=probs)
        
        if new_cluster == self.K:
            best_idx = np.argmax(log_liks)
            self.kernels[self.K] = kernel_samples[best_idx]
            self.K += 1
            
        return new_cluster
    
    def _cleanup_empty_clusters(self):
        active_clusters = torch.unique(self.z)     
        if len(active_clusters) < self.K:
            cluster_map = {old.item(): new for new, old in enumerate(active_clusters)}
            z_new = torch.zeros_like(self.z) 
            for old, new in cluster_map.items():
                z_new[self.z == old] = new
            self.z = z_new
            new_kernels = {new: self.kernels[old] for old, new in cluster_map.items()}
            self.kernels = new_kernels
            self.K = len(active_clusters)
            
    def _log_prior_kernel(self, kernel):
        sigma_f = kernel.sigma_f       
        length_scale = kernel.length_scale  
        sigma_n = kernel.sigma_n    

        log_p = InverseGamma(
            self.prior_params['sigma_f'][0],
            self.prior_params['sigma_f'][1]
        ).log_prob(sigma_f)

        log_p += InverseGamma(
            self.prior_params['length_scale'][0],
            self.prior_params['length_scale'][1]
        ).log_prob(length_scale)

        log_p += InverseGamma(
            self.prior_params['sigma_n'][0],
            self.prior_params['sigma_n'][1]
        ).log_prob(sigma_n)
        
        return log_p  
    
    def _log_marginal_likelihood(self, y, X, kernel):
        n = len(y) 
        K = kernel(X, X, noise=True)
        try:
            alpha, L, _ = cholesky_solve(K, y)
            data_fit = -0.5 * torch.dot(y, alpha)  
            complexity = -torch.sum(torch.log(torch.diag(L)))  
            constant = -0.5 * n * np.log(2 * np.pi)  
            log_lik = data_fit + complexity + constant 
            return log_lik
        except RuntimeError:
            return torch.tensor(-1e10, device=self.device)
            
    def _optimize_hyperparameters(self, k, n_steps=50, lr=0.01):
        mask = (self.z == k)
        if torch.sum(mask) == 0:
            return  
        
        X_k = self.X[mask]  
        y_k = self.y[mask] 
        
        kernel = self.kernels[k]

        optimizer = optim.Adam(kernel.parameters(), lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            log_lik = self._log_marginal_likelihood(y_k, X_k, kernel)  
            log_prior = self._log_prior_kernel(kernel) 
            loss = -(log_lik + log_prior) 
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            
    def _total_log_likelihood(self):
        total = 0.0
        for k in range(self.K):
            mask = (self.z == k)  
            if torch.sum(mask) > 0:
                log_lik = self._log_marginal_likelihood(
                    self.y[mask],  
                    self.X[mask], 
                    self.kernels[k]
                )
                total += log_lik.item()  
        return total
        
    def _print_cluster_summary(self):
        print("\nCluster Summary:")
        print("-" * 70)
        for k in range(self.K):
            n_k = torch.sum(self.z == k).item()
            params = self.kernels[k].get_params_dict()
            print(f"Cluster {k}: {n_k:3d} points | "
                  f"sigma_f={params['sigma_f']:6.3f}, "
                  f"length_scale={params['length_scale']:6.3f}, "
                  f"sigma_n={params['sigma_n']:6.3f}")
            
    def fit(self, X, y, n_iterations=1000, burn_in=500, optim_every=5, verbose=True):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1)

        self.x_mean = X.mean(dim=0)                                           
        self.x_std  = X.std(dim=0).clamp_min(1e-12)
        self.y_mean = y.mean()
        self.y_std  = y.std().clamp_min(1e-12)

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        self.X = Xn
        self.y = yn
        self.n = len(self.y)

        print(f"Training data (normalized): X shape = {self.X.shape}, y shape = {self.y.shape}")

        self._initialize_clusters()
        print(f"Initialized with K={self.K} clusters")

        for iteration in tqdm(range(n_iterations)):
            perm = torch.randperm(self.n, device=self.device)

            for i in perm:
                self.z[i.item()] = self._sample_cluster_assignment(i.item())
            self._cleanup_empty_clusters()

            if (iteration + 1) % optim_every == 0:
                for k in range(self.K):
                    self._optimize_hyperparameters(k, n_steps=30, lr=0.01)

            if iteration >= burn_in:
                self.samples_z.append(self.z.cpu().clone())  # Store on CPU to save GPU memory
                self.samples_K.append(self.K)

            log_lik = self._total_log_likelihood()
            self.log_likelihoods.append(log_lik)

            if (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}/{n_iterations}, "
                      f"K={self.K}, "
                      f"LogLik={log_lik:.2f}")

        print(f"\nFinal model: {self.K} clusters")
        self._print_cluster_summary()

        return self

    @torch.no_grad()
    def predict(self, X_test, return_std=True, return_components=False):

        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        n_test = len(X_test)

        X_test_norm = (X_test - self.x_mean) / self.x_std

        cluster_probs = torch.zeros(n_test, self.K, device=self.device)  

        for j in range(n_test):
            x_test = X_test_norm[j:j+1]
            log_probs = []

            for k in range(self.K):
                mask = (self.z == k) 
                X_k = self.X[mask]
                y_k = self.y[mask]
                n_k = torch.sum(mask).item()
                log_prior = np.log(n_k / (self.n + self.alpha))

                mu_pred, var_pred = self._gp_predict(x_test, X_k, y_k, self.kernels[k])
                log_spatial = -0.5 * torch.log(var_pred[0] + 1e-6) 
                log_probs.append(log_prior + log_spatial.item())

            log_probs = np.array(log_probs) 
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)  

            cluster_probs[j] = torch.tensor(probs, device=self.device)  

        y_pred_n = torch.zeros(n_test, device=self.device)
        y_var_n  = torch.zeros(n_test, device=self.device)

        mu_list_n = []
        var_list_n = []

        for k in range(self.K):
            mask = (self.z == k)
            X_k = self.X[mask]
            y_k = self.y[mask]

            mu_k_n, var_k_n = self._gp_predict(X_test_norm, X_k, y_k, self.kernels[k])
            if return_components:
                mu_list_n.append(mu_k_n)
                var_list_n.append(var_k_n)

            y_pred_n += cluster_probs[:, k] * mu_k_n
            y_var_n  += cluster_probs[:, k] * (var_k_n + mu_k_n**2)

        y_var_n -= y_pred_n**2
        y_var_n = torch.clamp(y_var_n, min=0)
        y_std_n = torch.sqrt(y_var_n) 

        y_pred = (y_pred_n * self.y_std + self.y_mean).cpu().numpy()
        y_std  = (y_std_n  * self.y_std).cpu().numpy()

        if return_components:
            w_k = cluster_probs.cpu().numpy()  # [N,K]
            mu_k = torch.stack(mu_list_n, dim=1) * self.y_std + self.y_mean
            std_k = torch.sqrt(torch.stack(var_list_n, dim=1)) * self.y_std
            mu_k = mu_k.cpu().numpy()
            std_k = std_k.cpu().numpy()
            return y_pred, y_std, w_k, mu_k, std_k

        if return_std:
            return y_pred, y_std
        else:
            return y_pred
        
    def predict_torch(self, X_test, return_std=True, return_components=False):
        device = self.device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        N = X_test.shape[0]

        Xn = (X_test - self.x_mean) / self.x_std

        counts = torch.stack([(self.z == k).sum() for k in range(self.K)], dim=0).to(torch.float32)
        eps = torch.finfo(torch.float32).tiny
        log_prior = torch.log((counts + eps) / (self.n + self.alpha))

        mu_k_list, var_k_list = [], []
        for k in range(self.K):
            mask = (self.z == k)
            X_k = self.X[mask]
            y_k = self.y[mask]
            mu_k_n, var_k_n = self._gp_predict(Xn, X_k, y_k, self.kernels[k])
            mu_k_list.append(mu_k_n)
            var_k_list.append(var_k_n)
        mu_k_n = torch.stack(mu_k_list, dim=1)
        var_k_n = torch.stack(var_k_list, dim=1).clamp_min(1e-12)

        log_spatial = -0.5 * torch.log(var_k_n + 1e-6)
        logits = log_spatial + log_prior.view(1, -1)
        w_k = torch.softmax(logits, dim=1)

        y_pred_n = (w_k * mu_k_n).sum(dim=1)
        y_m2_n   = (w_k * (var_k_n + mu_k_n**2)).sum(dim=1)
        y_var_n  = (y_m2_n - y_pred_n**2).clamp_min(0.0)
        y_std_n  = torch.sqrt(y_var_n)

        y_pred = y_pred_n * self.y_std + self.y_mean
        y_std  = y_std_n  * self.y_std

        if return_components:
            mu_k = mu_k_n * self.y_std + self.y_mean
            std_k = torch.sqrt(var_k_n).clamp_min(0.0) * self.y_std
            if return_std:
                return y_pred, y_std, w_k, mu_k, std_k
            else:
                return y_pred, None, w_k, mu_k, std_k

        if return_std:
            return y_pred, y_std
        else:
            return y_pred
