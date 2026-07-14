import torch
import math
import numpy as np
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
    beta: float = 1.5                   # Used strictly for the Fefferman minimax radius scaling



def construct_whitney_atlas(data: torch.Tensor, 
                            intrinsic_dim: int,
                            empirical_config: EmpiricalConfig = None) -> tuple:
    
    # 0. Initialize default empirical configuration if none provided
    if empirical_config is None:
        empirical_config = EmpiricalConfig()
        
    N, p = data.shape
    d = int(intrinsic_dim)
    device = data.device
    
    # 1. APPLY VOLUME SCALE TO MINIMAX RADIUS
    # delta_minimax = c * n^(-1 / (2*beta + d))
    delta_minimax = empirical_config.volume_scale * math.pow(N, -1.0 / (2.0 * empirical_config.beta + float(d)))
    
    # 2. A. Fefferman implementation (Farthest Point Sampling)
    initial_idx = int(torch.randint(0, N, (1,), device=device).item())
    fps_indices = [initial_idx]
    
    # Initialize the minimum distance tracker using pure ambient Euclidean distance (R^p)
    min_ambient_distances = torch.cdist(data, data[initial_idx:initial_idx+1]).squeeze(1)
    
    while True:
        # Evaluate the maximum unassigned distance in the ambient space
        max_dist, farthest_idx = torch.max(min_ambient_distances, dim=0)
        
        # The Minimax Halting Gate: 
        # Halt strictly when the maximal distance drops below the delta bound.
        if max_dist.item() <= delta_minimax + 1e-7:
            break
            
        new_center_idx = farthest_idx.item()
        fps_indices.append(new_center_idx)
        
        # Dynamically update the minimum ambient distance tensor
        dist_to_new_center = torch.cdist(data, data[new_center_idx:new_center_idx+1]).squeeze(1)
        min_ambient_distances = torch.minimum(min_ambient_distances, dist_to_new_center)

    # 3. CHART ASSIGNMENT VIA DUAL-CONDITION BANDWIDTHS
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

        # 4. BASE TOPOLOGICAL FOUNDATION (1st-Order PCA)
        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i

        # 1st-Order Tangent Space
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
        
        atlas_frames.append(frame_data)

    global_membership_mask = torch.cat(membership_masks, dim=1).cpu()


    return global_membership_mask, atlas_frames, intrinsic_coords, chart_ambient_indices, fps_centers