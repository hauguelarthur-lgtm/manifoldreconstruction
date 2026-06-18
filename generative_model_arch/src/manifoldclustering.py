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
    
    # Compute empirical Euclidean barycenters for downstream references
    centers = []
    for i in range(num_charts):
        mask = (labels == i)
        centers.append(data[mask].mean(dim=0))
    cluster_centers = torch.stack(centers)
    
    return labels, cluster_centers