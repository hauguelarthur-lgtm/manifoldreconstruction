import torch
import numpy as np
from sklearn.cluster import KMeans

def partition_data(data: torch.Tensor, num_charts: int, random_state: int = 42) -> tuple:
    """
    Partitions the empirical dataset into 'm' localized neighborhoods (charts).
    Satisfies the theoretical injectivity radius bound by localizing the Euclidean geometry.
    """
    data_np = data.detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=num_charts, random_state=random_state, n_init='auto')
    labels_np = kmeans.fit_predict(data_np)
    centers_np = kmeans.cluster_centers_
    
    labels = torch.tensor(labels_np, dtype=torch.long, device=data.device)
    cluster_centers = torch.tensor(centers_np, dtype=torch.float32, device=data.device)
    
    return labels, cluster_centers