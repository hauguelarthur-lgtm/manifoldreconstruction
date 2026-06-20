import torch
import numpy as np
from sklearn.cluster import SpectralClustering

def partition_data(data: torch.Tensor, num_charts: int, random_state: int = 42) -> tuple:
    data_np = data.detach().cpu().numpy()
    
    # K-NN affinity enforces intrinsic manifold geometry
    clustering = SpectralClustering(
        n_clusters=num_charts, 
        affinity='nearest_neighbors',
        n_neighbors=15,
        random_state=random_state
    )
    labels_np = clustering.fit_predict(data_np)
    labels = torch.tensor(labels_np, dtype=torch.long, device=data.device)
    
    p = data.shape[1]
    centers = []
    precisions = []
    
    # Compute empirical Euclidean barycenters and Anisotropic Precision Matrices
    for i in range(num_charts):
        mask = (labels == i)
        chart_data = data[mask]
        
        mu = chart_data.mean(dim=0)
        centers.append(mu)
        
        # Compute empirical covariance matrix
        centered_data = chart_data - mu
        cov = torch.matmul(centered_data.T, centered_data) / (chart_data.size(0) - 1)
        
        # Inject Tikhonov stabilization to guarantee invertibility of the zero-subspace
        reg_cov = cov + torch.eye(p, device=data.device) * 1e-4
        
        # Precision matrix is the inverse of the covariance matrix
        precisions.append(torch.linalg.inv(reg_cov))
        
    cluster_centers = torch.stack(centers)
    cluster_precisions = torch.stack(precisions)
    
    return labels, cluster_centers, cluster_precisions