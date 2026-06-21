import torch

def compute_smooth_partition_of_unity(x: torch.Tensor, 
                                      cluster_centers: torch.Tensor, 
                                      precisions: torch.Tensor = None,
                                      length_scale: float = 1.0) -> torch.Tensor:
    """
    Computes a smooth Gaussian partition of unity across ambient chart centroids.
    Incorporates Mahalanobis precision matrices to adapt to local chart covariances.
    """
    batch_size, p_dim = x.shape
    m = cluster_centers.shape[0]
    
    if precisions is None:
        distances_sq = torch.cdist(x, cluster_centers) ** 2
    else:
        distances_sq = torch.zeros(batch_size, m, device=x.device)
        for i in range(m):
            delta = x - cluster_centers[i] 
            dist_i = torch.sum(torch.matmul(delta, precisions[i]) * delta, dim=1)
            distances_sq[:, i] = dist_i

    # Scale logits invariant to ambient dimension p to prevent softmax hard-saturation
    adjusted_scale = length_scale * torch.sqrt(torch.tensor(p_dim, dtype=torch.float32, device=x.device))
    logits = -distances_sq / (2 * adjusted_scale ** 2)
    
    return torch.softmax(logits, dim=1)

def apply_terminal_ambient_gluing(X_lifted: torch.Tensor,
                                  whitney_atlas: list,
                                  cluster_centers: torch.Tensor,
                                  cluster_precisions: torch.Tensor,
                                  length_scale: float = 1.0) -> torch.Tensor:
    """
    Terminal post-processing: Blends overlapping Whitney chart boundaries in ambient space R^p.
    Evaluates the partition of unity on lifted coordinates to smooth multi-chart transitions.
    """
    weights = compute_smooth_partition_of_unity(X_lifted, cluster_centers, cluster_precisions, length_scale)
    X_glued = torch.zeros_like(X_lifted)
    m = len(whitney_atlas)
    
    for i in range(m):
        mu_i = whitney_atlas[i]['mu'].to(X_lifted.device)
        Q_i = whitney_atlas[i]['Q'].to(X_lifted.device)
        
        # Local chart projection in ambient space: P_i(x) = (x - mu_i) @ Q_i @ Q_i.T + mu_i
        delta = X_lifted - mu_i
        proj_i = torch.matmul(torch.matmul(delta, Q_i), Q_i.T) + mu_i
        
        rho_i = weights[:, i].unsqueeze(1)
        X_glued += rho_i * proj_i
        
    return X_glued