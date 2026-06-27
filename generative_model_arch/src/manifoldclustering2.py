import torch
import math
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from itertools import combinations_with_replacement
from dataclasses import dataclass
from typing import Optional
import scipy.sparse as sp

@dataclass
class EmpiricalConfig:
    """
    Exposes the hidden constants within the asymptotic minimax bounds 
    to calibrate the algorithm for specific real-world datasets.
    """
    volume_scale: float = 1.0           # Multiplies delta_minimax to match data spread
    oversample_ratio: float = 1.5       # Safety margin for degrees of freedom (e.g., 1.5x)
    lambda_base: float = 1e-7           # Absolute floor for Tikhonov regularization
    lambda_trace_scale: float = 1e-4    # Dynamic trace-scaling factor for Tikhonov regularization
    max_radius_cap: Optional[float] = None # Absolute ceiling to prevent infinite expansion in outliers

def get_poly_features(U: torch.Tensor, k_degree: int) -> torch.Tensor:
    """
    Dynamically generates all symmetric polynomial feature combinations 
    from degree 2 up to k_degree.
    """
    d = U.shape[1]
    features = []
    for degree in range(2, k_degree + 1):
        for combo in combinations_with_replacement(range(d), degree):
            # Compute the Hadamard product of the selected intrinsic coordinate columns
            term = U[:, combo[0]]
            for idx in combo[1:]:
                term = term * U[:, idx]
            features.append(term.unsqueeze(1))
            
    if not features:
        return torch.empty((U.shape[0], 0), device=U.device)
    return torch.cat(features, dim=1)

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
        mask = dist_sq < r_sq
        weights = torch.zeros_like(dist_sq)
        normalized_sq = dist_sq[mask] / r_sq
        weights[mask] = torch.exp(-1.0 / (1.0 - normalized_sq))
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
                p_local += torch.matmul(u_poly, W.T)

            x_proj += w * p_local
            weight_sum += w

        valid_mask = weight_sum > 0
        x_proj[valid_mask.squeeze()] /= weight_sum[valid_mask]
        
        return x_proj

def construct_whitney_atlas(data: torch.Tensor, 
                            intrinsic_dim: int,
                            target_beta: float = 1.50,
                            k_neighbors: int = 15,
                            empirical_config: EmpiricalConfig = None) -> tuple:
    
    # 0. Initialize default empirical configuration if none provided
    if empirical_config is None:
        empirical_config = EmpiricalConfig()
        
    N, p = data.shape
    d = int(intrinsic_dim)
    device = data.device
    k_degree = math.floor(target_beta) + 1

    # 1. APPLY VOLUME SCALE TO MINIMAX RADIUS
    # delta_minimax = c * n^(-1 / (2*beta + d))
    delta_minimax = empirical_config.volume_scale * math.pow(N, -1.0 / (2.0 * target_beta + float(d)))
    
    # 2. APPLY OVERSAMPLING RATIO TO DEGREES OF FREEDOM
    poly_dim = 0
    if k_degree >= 2:
        poly_dim = sum(math.comb(d + j - 1, j) for j in range(2, k_degree + 1))
    
    # Safely pad the required points to prevent noise-induced singularities
    min_points_required = int(math.ceil((poly_dim + d) * empirical_config.oversample_ratio))

    # 3. RADIUS-DRIVEN INTRINSIC GEODESIC FPS
    np_data = data.cpu().numpy()
    
    # Base parameters for the fluctuating k-NN
    k_max = min(50, N - 1)  # Absolute upper bound of connections evaluated
    local_scale_neighbor = min(5, k_max) # The neighbor used to measure local density (sigma)
    tau_reach_limit = 2.5 # Maximum allowed distance multiplier before edge is cut
    
    # Query the maximum possible neighborhood
    nbrs = NearestNeighbors(n_neighbors=k_max, algorithm='auto').fit(np_data)
    distances, indices = nbrs.kneighbors(np_data)
    
    row_indices = []
    col_indices = []
    edge_weights = []
    
    # Dynamically fluctuate k_i based on local density
    for i in range(N):
        # Sigma_i is the distance to the m-th closest neighbor (local density proxy)
        sigma_i = distances[i, local_scale_neighbor - 1] 
        
        # The adaptive cutoff threshold for point i
        adaptive_threshold = sigma_i * tau_reach_limit
        
        for j_idx, neighbor_dist in enumerate(distances[i]):
            if j_idx == 0: 
                continue # Skip self-loop
                
            neighbor_index = indices[i, j_idx]
            
            # FLUCTUATION LOGIC: Only keep the edge if it respects the local density bound.
            # In dense areas, many points pass. In sparse areas, very few pass.
            if neighbor_dist <= adaptive_threshold:
                row_indices.append(i)
                col_indices.append(neighbor_index)
                edge_weights.append(neighbor_dist)
                
    # Build the sparse adjacency matrix for Dijkstra
    knn_graph = sp.csr_matrix((edge_weights, (row_indices, col_indices)), shape=(N, N))
    
    # Ensure symmetry (undirected graph) so paths can travel both ways
    knn_graph = knn_graph.maximum(knn_graph.T)
    
    fps_indices = [int(torch.randint(0, N, (1,)).item())]
    geodesic_distances = dijkstra(knn_graph, indices=fps_indices[0], directed=False)
    
    while True:
        valid_distances = geodesic_distances[geodesic_distances != np.inf]
        if len(valid_distances) == 0:
            break
            
        max_dist = float(np.max(valid_distances))
        if max_dist <= delta_minimax:
            break
            
        farthest_idx = np.argmax(geodesic_distances)
        fps_indices.append(farthest_idx)
        
        dist_to_new = dijkstra(knn_graph, indices=farthest_idx, directed=False)
        geodesic_distances = np.minimum(geodesic_distances, dist_to_new)

    # 4. CHART ASSIGNMENT VIA DUAL-CONDITION BANDWIDTHS
    m = len(fps_indices)
    fps_centers = data[fps_indices]
    all_pairwise_dists = torch.cdist(data, fps_centers) 

    atlas_frames = []
    intrinsic_coords = []
    chart_ambient_indices = []
    membership_masks = []

    for i in range(m):
        dists_to_center = all_pairwise_dists[:, i]
        
        kth_dist = torch.kthvalue(dists_to_center, min_points_required).values.item()
        resolved_radius = max(delta_minimax * 1.5, kth_dist * 1.1)
        
        in_chart_mask = dists_to_center < resolved_radius
        membership_masks.append(in_chart_mask.unsqueeze(1))
        
        chart_idx = torch.nonzero(in_chart_mask).squeeze(1)
        chart_ambient_indices.append(chart_idx.cpu())

        X_i = data[chart_idx]
        N_i = X_i.size(0)

        # 5. TAYLOR REGRESSION
        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i

        # 1st-Order Tangent Space
        cov_i = torch.matmul(centered_X.T, centered_X) / (N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        top_indices = torch.argsort(eigenvalues, descending=True)[:d]
        Q_i = eigenvectors[:, top_indices]  

        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        frame_data = {
            'mu': mu_i.cpu(), 
            'Q': Q_i.cpu(),
            'r_sq': resolved_radius ** 2
        }

        # Adaptive Higher-Order Expansion
        if k_degree >= 2 and N_i >= poly_dim + d:
            N_err = centered_X - torch.matmul(U_i, Q_i.T)  
            U_poly = get_poly_features(U_i, k_degree)

            U_poly_mean = U_poly.mean(dim=0, keepdim=True)
            U_poly_std = U_poly.std(dim=0, keepdim=True) + 1e-8
            U_poly_norm = (U_poly - U_poly_mean) / U_poly_std

            global_n_err_std = torch.sqrt(torch.sum(torch.var(N_err, dim=0)) / float(p - d)) + 1e-6
            N_err_norm = N_err / global_n_err_std

            G = torch.matmul(U_poly_norm.T, U_poly_norm)
            
            # 4. APPLY EMPIRICAL TIKHONOV NOISE REGULARIZATION
            lambda_reg = (empirical_config.lambda_trace_scale * (torch.trace(G) / poly_dim) 
                          + empirical_config.lambda_base)
            
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

    return global_membership_mask, intrinsic_coords, global_manifold, chart_ambient_indices