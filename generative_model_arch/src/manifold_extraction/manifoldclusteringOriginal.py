import torch
import math
import numpy as np
from itertools import combinations_with_replacement
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmpiricalConfig:
    """
    Exposes the hidden constants within the asymptotic minimax bounds 
    to calibrate the algorithm for specific real-world datasets.
    """
    volume_scale: float = 0.2           # Multiplies delta_minimax to match data spread
    C_overlap: float = 1.5
    beta:float=1.5

def get_poly_features(U: torch.Tensor, k_degree: int) -> torch.Tensor:
    """
    Generates symmetric polynomial feature combinations from degree 2 up to k_degree
    using dynamic programming to eliminate redundant sub-expression computation.
    """
    # Guard against 1D tensors and mathematically empty polynomial degrees
    if k_degree < 2 or U.ndim != 2:
        return torch.empty((U.shape[0], 0), device=U.device)
        
    N, d = U.shape
    all_features = []
    
    # Initialize Base Case (Degree 1)
    # Stored as tuples of (tensor_column, index) to track valid multiplication paths.
    # U[:, i:i+1] natively keeps the shape as (N, 1) without requiring unsqueeze.
    prev_terms = [(U[:, i:i+1], i) for i in range(d)]
    
    for _ in range(2, k_degree + 1):
        curr_terms = []
        for term, last_idx in prev_terms:
            # Multiply only by features with index >= last_idx to maintain symmetry
            # and prevent generating permutations of the same combination.
            for i in range(last_idx, d):
                new_term = term * U[:, i:i+1]
                curr_terms.append((new_term, i))
                all_features.append(new_term)
                
        # Advance the memoization state for the next polynomial degree
        prev_terms = curr_terms
        
    if not all_features:
        return torch.empty((N, 0), device=U.device)
        
    # Execute a single memory reallocation and copy
    return torch.cat(all_features, dim=1)


class WhitneyPartitionOfUnity:
    """
    Evaluates the globally C^{\beta+1}-smooth manifold approximation.
    Upgraded to dynamically evaluate generalized multi-order Taylor polynomials.
    """
    def __init__(self, atlas_frames: list[dict], k_degree: int, device: torch.device):
        self.atlas = atlas_frames
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

            # Intrinsic projection (Degree 1)
            u = torch.matmul(x - mu, Q)
            p_local = mu + torch.matmul(u, Q.T)
            
            # Evaluate Generalized Higher-Order Taylor Polynomial
            if 'W' in frame and self.k_degree >= 2:
                W = frame['W'].to(self.device)
                
                # Reconstruct exact tensor combinations up to k_degree
                u_poly = get_poly_features(u, self.k_degree)
                
                # Add higher-order normal bundle shift
                p_local += torch.matmul(u_poly, W)

            x_proj += w * p_local
            weight_sum += w

        valid_mask = weight_sum > 0
        x_proj[valid_mask.squeeze()] /= weight_sum[valid_mask.squeeze()]
        
        return x_proj

def construct_whitney_atlas(data: torch.Tensor, 
                            intrinsic_dim: int,
                            empirical_config: EmpiricalConfig = None) -> tuple:
    
    # 0. Initialize default empirical configuration if none provided
    if empirical_config is None:
        empirical_config = EmpiricalConfig()
        
    N, p = data.shape
    d = int(intrinsic_dim)
    device = data.device
    k_degree = math.floor(empirical_config.beta) + 1
    # 1. APPLY VOLUME SCALE TO MINIMAX RADIUS
    # delta_minimax = c * n^(-1 / (2*beta + d))
    delta_minimax = empirical_config.volume_scale * math.pow(N, -1.0 / (2.0 * empirical_config.beta + float(d)))
    
    # 2. APPLY OVERSAMPLING RATIO TO DEGREES OF FREEDOM

    poly_dim = 0

    if k_degree >= 2:   
        # Calculate the exact algebraic degrees of freedom required for the Taylor polynomial tensor
        poly_dim = sum(math.comb(d + j - 1, j) for j in range(2, k_degree + 1))


    # 3. A.Feerman implementation
    
    initial_idx = int(torch.randint(0, N, (1,), device=device).item())
    fps_indices = [initial_idx]
    
    # Initialize the minimum distance tracker using pure ambient Euclidean distance (R^p)
    # The slice data[initial_idx:initial_idx+1] maintains the 2D shape required by torch.cdist
    min_ambient_distances = torch.cdist(data, data[initial_idx:initial_idx+1]).squeeze(1)
    
    while True:
        # Evaluate the maximum unassigned distance in the ambient space
        max_dist, farthest_idx = torch.max(min_ambient_distances, dim=0)
        
        # The Minimax Halting Gate: 
        # Halt strictly when the maximal distance drops below the delta bound.
        # This guarantees the set is mathematically delta-separated and delta-dense.
        if max_dist.item() <= delta_minimax + 1e-7:
            break
            
        new_center_idx = farthest_idx.item()
        fps_indices.append(new_center_idx)
        
        # Dynamically update the minimum ambient distance tensor
        dist_to_new_center = torch.cdist(data, data[new_center_idx:new_center_idx+1]).squeeze(1)
        min_ambient_distances = torch.minimum(min_ambient_distances, dist_to_new_center)

    

    

    # 4. CHART ASSIGNMENT VIA DUAL-CONDITION BANDWIDTHS
    m = len(fps_indices)
    fps_centers = data[fps_indices]

    atlas_frames = []
    intrinsic_coords = []
    chart_ambient_indices = []
    membership_masks = []

    resolved_radius = delta_minimax * empirical_config.C_overlap
    for i in range(m):
        dists_to_center = torch.cdist(data, fps_centers[i:i+1]).squeeze(1)
        
        in_chart_mask = dists_to_center < resolved_radius
        membership_masks.append(in_chart_mask.unsqueeze(1))
        
        chart_idx = torch.nonzero(in_chart_mask).squeeze(1)
        chart_ambient_indices.append(chart_idx.cpu())

        X_i = data[chart_idx]
        N_i = X_i.size(0)

        # 5. TAYLOR REGRESSION
        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i
 
        # 1st-Order Tangent Space (PCA)
        cov_i = torch.matmul(centered_X.T, centered_X) / max(1, N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        top_indices = torch.argsort(eigenvalues, descending=True)[:d]
        Q_i = eigenvectors[:, top_indices]  

        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        frame_data = {
            'mu': mu_i, 
            'Q': Q_i,
            'r_sq': resolved_radius ** 2
        }

        # Adaptive Higher-Order Expansion
        if k_degree >= 2 and N_i >= poly_dim + d:
            N_err = centered_X - torch.matmul(U_i, Q_i.T)  
            U_poly = get_poly_features(U_i, k_degree)

            U_poly_mean = U_poly.mean(dim=0, keepdim=True)
            U_poly_std = U_poly.std(dim=0, keepdim=True) + 1e-8
            U_poly_norm = (U_poly - U_poly_mean) / U_poly_std

            global_n_err_std = torch.sqrt(torch.sum(torch.var(N_err, dim=0)) / float(p)) + 1e-6
            N_err_norm = N_err / global_n_err_std

            G = torch.matmul(U_poly_norm.T, U_poly_norm)
            
            # 4. APPLY EMPIRICAL TIKHONOV NOISE REGULARIZATION
            # Empirical Tikhonov Law: Decouples lambda from Grid Search
            # 1. Base floor locked strictly to machine floating-point precision
            lambda_base = 1e-8 if U_poly.dtype == torch.float32 else 1e-14

            # 2. Trace scale scales dynamically with local finite-sample deficiency
            # If N_i >> poly_dim, damping goes to near-zero (pure geometric fit).
            # If N_i approaches poly_dim, damping scales up to prevent inversion blow-up.
            sample_density_ratio = float(poly_dim) / max(float(N_i), 1.0)
            empirical_trace_scale = max(1e-6, sample_density_ratio ** 2)

            lambda_reg = empirical_trace_scale * (torch.trace(G) / poly_dim) + lambda_base
            G_reg = G + torch.eye(poly_dim, device=device) * lambda_reg
            rhs = torch.matmul(U_poly_norm.T, N_err_norm)
            
            try:
                W_norm = torch.linalg.solve(G_reg, rhs)
            except (RuntimeError, torch._C._LinAlgError):
                W_norm = torch.linalg.pinv(G_reg, hermitian=True).matmul(rhs)

            W_i = (W_norm / U_poly_std.T) * global_n_err_std
            
            # Multi-order intercept alignment
            mu_i_star = mu_i - torch.matmul(U_poly_mean.squeeze(0), W_i)
            
            frame_data['mu'] = mu_i_star.cpu()
            frame_data['W'] = W_i.cpu()

        atlas_frames.append(frame_data)

    global_membership_mask = torch.cat(membership_masks, dim=1).cpu()

    # 6. INITIALIZE GLOBAL GLUING MECHANISM
    # Pass k_degree to the unified partition to ensure isomorphic polynomial reconstruction
    global_manifold = WhitneyPartitionOfUnity(atlas_frames, k_degree, device)

    return global_membership_mask, intrinsic_coords, global_manifold, chart_ambient_indices, fps_centers