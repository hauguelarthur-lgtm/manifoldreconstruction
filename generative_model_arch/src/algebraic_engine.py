import torch
import math

def get_poly_features(U: torch.Tensor, k_degree: int) -> torch.Tensor:
    """
    Generates symmetric polynomial feature combinations from degree 2 up to k_degree
    using dynamic programming to eliminate redundant sub-expression computation.
    """
    if k_degree < 2 or U.ndim != 2:
        return torch.empty((U.shape[0], 0), device=U.device)
        
    N, d = U.shape
    all_features = []
    
    prev_terms = [(U[:, i:i+1], i) for i in range(d)]
    
    for _ in range(2, k_degree + 1):
        curr_terms = []
        for term, last_idx in prev_terms:
            for i in range(last_idx, d):
                new_term = term * U[:, i:i+1]
                curr_terms.append((new_term, i))
                all_features.append(new_term)
                
        prev_terms = curr_terms
        
    if not all_features:
        return torch.empty((N, 0), device=U.device)
        
    return torch.cat(all_features, dim=1)

def apply_algebraic_taylor_regression(
    data_ambient: torch.Tensor, 
    chart_ambient_indices: list[torch.Tensor], 
    atlas_frames: list[dict], 
    d: int, 
    beta: float,
    device: torch.device
) -> list[dict]:
    """
    Ingests 1st-order topological frames and computes regularized higher-order 
    Taylor polynomial curvature tensors W_i via the Empirical Tikhonov Law.
    """
    k_degree = math.floor(beta) + 1
    p = data_ambient.size(1)
    
    poly_dim = sum(math.comb(d + j - 1, j) for j in range(2, k_degree + 1))
    augmented_atlas = []

    for i in range(len(atlas_frames)):
        frame = atlas_frames[i].copy()
        
        if k_degree < 2:
            augmented_atlas.append(frame)
            continue
            
        chart_idx = chart_ambient_indices[i]
        X_i = data_ambient[chart_idx]
        N_i = X_i.size(0)
        
        mu_i = frame['mu'].to(device)
        Q_i = frame['Q'].to(device)
        
        centered_X = X_i - mu_i
        U_i = torch.matmul(centered_X, Q_i)

        if N_i >= poly_dim + d:
            # 1. Extract the Normal Bundle Residual
            N_err = centered_X - torch.matmul(U_i, Q_i.T)  
            
            # 2. Compute symmetric polynomial combinations
            U_poly = get_poly_features(U_i, k_degree)

            U_poly_mean = U_poly.mean(dim=0, keepdim=True)
            U_poly_std = U_poly.std(dim=0, keepdim=True) + 1e-8
            U_poly_norm = (U_poly - U_poly_mean) / U_poly_std

            global_n_err_std = torch.sqrt(torch.sum(torch.var(N_err, dim=0)) / float(p)) + 1e-6
            N_err_norm = N_err / global_n_err_std

            # 3. Construct Empirical Gram Matrix
            G = torch.matmul(U_poly_norm.T, U_poly_norm)
            
            # 4. Empirical Tikhonov Noise Regularization
            lambda_base = 1e-8 if U_poly.dtype == torch.float32 else 1e-14
            sample_density_ratio = float(poly_dim) / max(float(N_i), 1.0)
            empirical_trace_scale = max(1e-6, sample_density_ratio ** 2)

            lambda_reg = empirical_trace_scale * (torch.trace(G) / poly_dim) + lambda_base
            G_reg = G + torch.eye(poly_dim, device=device) * lambda_reg
            
            rhs = torch.matmul(U_poly_norm.T, N_err_norm)
            
            try:
                W_norm = torch.linalg.solve(G_reg, rhs)
            except (RuntimeError, torch._C._LinAlgError):
                W_norm = torch.linalg.pinv(G_reg, hermitian=True).matmul(rhs)

            # 5. Reverse normalization scaling
            W_i = (W_norm / U_poly_std.T) * global_n_err_std
            
            # 6. Multi-order intercept alignment
            mu_i_star = mu_i - torch.matmul(U_poly_mean.squeeze(0), W_i)
            
            # Inject curvature tensors into the frame
            frame['mu'] = mu_i_star.cpu()
            frame['W'] = W_i.cpu()

        augmented_atlas.append(frame)

    return augmented_atlas

class AlgebraicWhitneyEvaluator:
    """
    Evaluates the globally C^{\beta+1}-smooth manifold approximation 
    strictly utilizing the pre-computed algebraic Taylor polynomial weights.
    """
    def __init__(self, augmented_atlas: list[dict], k_degree: int, device: torch.device):
        self.atlas = augmented_atlas
        self.k_degree = k_degree
        self.device = device

    def _bump_function(self, dist_sq: torch.Tensor, r_sq: float) -> torch.Tensor:
        normalized_sq = dist_sq / r_sq
        weights = torch.where(
            dist_sq < r_sq,
            torch.exp(-1.0 / (1.0 - normalized_sq)),
            torch.zeros_like(dist_sq)
        )
        return weights

    def evaluate_manifold(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.zeros_like(x)
        weight_sum = torch.zeros(x.size(0), 1, device=self.device)

        for frame in self.atlas:
            mu = frame['mu'].to(self.device)
            Q = frame['Q'].to(self.device)
            r_sq = frame['r_sq']
            
            dist_sq = torch.sum((x - mu) ** 2, dim=1, keepdim=True)
            w = self._bump_function(dist_sq, r_sq)

            # 1st-Order Intrinsic projection
            u = torch.matmul(x - mu, Q)
            p_local = mu + torch.matmul(u, Q.T)
            
            # Generalized Higher-Order Taylor Polynomial evaluation
            if 'W' in frame and self.k_degree >= 2:
                W = frame['W'].to(self.device)
                u_poly = get_poly_features(u, self.k_degree)
                p_local += torch.matmul(u_poly, W)

            x_proj += w * p_local
            weight_sum += w

        valid_mask = weight_sum > 0
        x_proj[valid_mask.squeeze()] /= weight_sum[valid_mask.squeeze()]
        
        return x_proj