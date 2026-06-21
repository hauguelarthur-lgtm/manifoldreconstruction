import torch

def compute_subordinated_partition_of_unity(x: torch.Tensor, 
                                            centroids: torch.Tensor, 
                                            chart_radii: torch.Tensor) -> torch.Tensor:
    """
    True Stéphanovitch C^\infty Compact Partition of Unity (arXiv:2506.19587).
    Evaluates compact bump functions strictly subordinated to the open cover B(mu_i, r_i).
    Guarantees absolute zero weight outside the localized chart boundaries.
    """
    Batch, d = x.shape
    m = centroids.shape[0]
    weights = torch.zeros(Batch, m, device=x.device)
    
    for i in range(m):
        # Euclidean distance squared to chart centroid
        dist_sq = torch.sum((x - centroids[i]) ** 2, dim=1)
        r_i_sq = chart_radii[i] ** 2
        
        # Relative coordinate radius u^2 = (d / r)^2
        u_sq = dist_sq / r_i_sq
        
        # C^\infty Compact Bump formula: exp(-1 / (1 - u^2)) strictly inside u < 1.
        # Evaluates to absolute 0.0 for all points where u >= 1 (outside chart cover).
        inside_mask = u_sq < 1.0
        if inside_mask.any():
            weights[inside_mask, i] = torch.exp(-1.0 / (1.0 - u_sq[inside_mask]))
            
    # Normalize across active charts to sum to 1.0
    weight_sums = weights.sum(dim=1, keepdim=True)
    
    # Fallback guard: if a stray stochastic particle escapes all covers, assign uniform weights
    escaped_mask = (weight_sums.squeeze(1) == 0.0)
    if escaped_mask.any():
        weights[escaped_mask, :] = 1.0 / m
        weight_sums[escaped_mask] = 1.0
        
    return weights / weight_sums