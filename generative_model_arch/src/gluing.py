import torch

def compute_ambient_subordinated_weights(X_ambient: torch.Tensor, 
                                         whitney_atlas: list, 
                                         covering_radius: float) -> torch.Tensor:
    """
    True Stéphanovitch C^\infty Compact Partition of Unity in Ambient Space R^p (arXiv:2506.19587).
    Evaluates compact bump functions exp(-1 / (1 - (dist/r)^2)) centered at ambient centroids \mu_i.
    """
    Batch, p = X_ambient.shape
    m = len(whitney_atlas)
    weights = torch.zeros(Batch, m, device=X_ambient.device)
    r_sq = covering_radius ** 2

    for i in range(m):
        mu_i = whitney_atlas[i]['mu'].to(X_ambient.device)
        dist_sq = torch.sum((X_ambient - mu_i) ** 2, dim=1)
        u_sq = dist_sq / r_sq
        
        inside_mask = u_sq < 1.0
        if inside_mask.any():
            weights[inside_mask, i] = torch.exp(-1.0 / (1.0 - u_sq[inside_mask]))

    weight_sums = weights.sum(dim=1, keepdim=True)
    escaped_mask = (weight_sums.squeeze(1) == 0.0)
    if escaped_mask.any():
        weights[escaped_mask, :] = 1.0 / m
        weight_sums[escaped_mask] = 1.0

    return weights / weight_sums

def apply_ambient_overlap_blending(X_ambient: torch.Tensor, 
                                   whitney_atlas: list, 
                                   covering_radius: float) -> torch.Tensor:
    """
    Blends overlapping ambient manifold patches using subordinated compact bump weights.
    Eliminates boundary shearing along chart transition seams.
    """
    weights = compute_ambient_subordinated_weights(X_ambient, whitney_atlas, covering_radius)
    X_blended = torch.zeros_like(X_ambient)
    m = len(whitney_atlas)

    for i in range(m):
        mu_i = whitney_atlas[i]['mu'].to(X_ambient.device)
        Q_i = whitney_atlas[i]['Q'].to(X_ambient.device)
        
        # Local tangent hyperplane projection in ambient space R^p
        delta = X_ambient - mu_i
        proj_i = torch.matmul(torch.matmul(delta, Q_i), Q_i.T) + mu_i
        
        rho_i = weights[:, i].unsqueeze(1)
        X_blended += rho_i * proj_i

    return X_blended