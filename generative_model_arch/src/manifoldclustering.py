import torch
import numpy as np
from sklearn.cluster import SpectralClustering

def construct_whitney_atlas(data: torch.Tensor, 
                            num_charts: int, 
                            intrinsic_dim: int, 
                            random_state: int = 42) -> tuple:
    """
    Constructs an intrinsic Whitney Submanifold Atlas.
    Partitions ambient data into local Euclidean charts, extracts their orthonormal 
    tangent frames via PCA, and projects coordinates into the intrinsic dimension 'd'.
    """
    # 1. K-NN affinity graph partitioning in ambient space
    data_np = data.detach().cpu().numpy()
    clustering = SpectralClustering(
        n_clusters=num_charts, 
        affinity='nearest_neighbors',
        n_neighbors=15,
        random_state=random_state
    )
    labels_np = clustering.fit_predict(data_np)
    labels = torch.tensor(labels_np, dtype=torch.long, device=data.device)
    
    p = data.shape[1]
    d = intrinsic_dim
    
    centers = []
    precisions = []
    atlas_frames = []
    intrinsic_coords = []
    
    for i in range(num_charts):
        mask = (labels == i)
        chart_data = data[mask]
        N_i = chart_data.size(0)
        
        # Structural guard against degenerate chart collapse
        if N_i <= d:
            raise ValueError(f"Chart {i} collapsed with only {N_i} samples. "
                             f"A minimum of d+1 ({d+1}) samples is mathematically required to span R^{d}.")
        
        # A. Empirical ambient centroid \mu_i \in R^p
        mu_i = chart_data.mean(dim=0)
        centers.append(mu_i)
        
        # B. Centered ambient chart data
        centered_data = chart_data - mu_i
        
        # C. Empirical covariance matrix \Sigma_i \in R^{p \times p}
        cov_i = torch.matmul(centered_data.T, centered_data) / (N_i - 1)
        
        # D. Ambient Precision via Moore-Penrose Pseudo-Inverse (saved for ambient gluing)
        precisions.append(torch.linalg.pinv(cov_i, rcond=1e-3))
        
        # E. Local Eigendecomposition to extract Whitney Tangent Frame Q_i \in R^{p \times d}
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        
        # Sort eigenvalues descending to isolate the top 'd' principal variance directions
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        top_indices = sorted_indices[:d]
        
        # Extract orthonormal basis Q_i (p x d)
        Q_i = eigenvectors[:, top_indices]
        atlas_frames.append({'mu': mu_i.cpu(), 'Q': Q_i.cpu()})
        
        # F. Pushforward ambient coordinates into local intrinsic Whitney chart U^{(i)} \in R^{N_i \times d}
        # Operation: (N_i x p) @ (p x d) -> (N_i x d)
        U_i = torch.matmul(centered_data, Q_i)
        intrinsic_coords.append(U_i)
        
    cluster_centers = torch.stack(centers)
    cluster_precisions = torch.stack(precisions)
    
    return labels, intrinsic_coords, atlas_frames, cluster_centers, cluster_precisions