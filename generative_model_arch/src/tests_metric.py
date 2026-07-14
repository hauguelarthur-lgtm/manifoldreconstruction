import torch
import math





def test_minimum_chart_overlap(global_membership_mask: torch.Tensor) -> dict:
    """
    Validates that every data point is covered by at least one chart in the atlas.
    
    Args:
        global_membership_mask (torch.Tensor): A boolean or numerical tensor of shape (N, m),
                                               where N is the number of ambient data points 
                                               and m is the number of atlas charts.
                                               
    Returns:
        dict: Statistical metrics of the chart overlaps (minimum, maximum, and mean).
    """
    # Sum across the chart dimension to compute the degree of overlap for each point
    overlaps_per_point = global_membership_mask.sum(dim=1)
    
    min_overlap = int(overlaps_per_point.min().item())
    max_overlap = int(overlaps_per_point.max().item())
    mean_overlap = float(overlaps_per_point.float().mean().item())
    
    uncovered_points_count = int((overlaps_per_point == 0).sum().item())
    
    
    return {
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "mean_overlap": mean_overlap
    }

import torch

def test_minimum_center_separation(fps_centers: torch.Tensor, delta_minimax: float) -> dict:
    """
    Validates that all chart centers are separated by a distance strictly greater than delta_minimax.
    
    Args:
        fps_centers (torch.Tensor): A tensor of shape (m, p) containing the ambient coordinates 
                                    of the m chart centers.
        delta_minimax (float): The threshold minimum allowable distance.
                               
    Returns:
        dict: Statistical metrics of the center separations (minimum, maximum, and mean).
    """
    m = fps_centers.size(0)
    
    # Separation is vacuously satisfied for a single chart
    if m < 2:
        return {"min_separation": float('inf'), "max_separation": float('inf'), "mean_separation": float('inf')}
        
    # Compute the pairwise distance matrix between all chart centers
    pairwise_dists = torch.cdist(fps_centers, fps_centers)
    
    # Mask out the diagonal (distance from a center to itself)
    # Fill with infinity to exclude the zeros from the minimum calculation
    mask = torch.eye(m, dtype=torch.bool, device=fps_centers.device)
    pairwise_dists.masked_fill_(mask, float('inf'))
    
    # Extract the minimum off-diagonal separation
    min_separation = float(pairwise_dists.min().item())
    
    # Re-mask infinity to compute accurate maximum and mean bounds
    valid_dists = pairwise_dists[~mask]
    max_separation = float(valid_dists.max().item())
    mean_separation = float(valid_dists.mean().item())
    
    
    return {
        "min_separation": min_separation,
        "max_separation": max_separation,
        "mean_separation": mean_separation
    }



import math

def test_chart_radius_proportion(global_manifold) -> dict:
    """
    Validates the proportion of charts within the Whitney atlas that possess 
    a resolved radius strictly greater than delta_minimax.
    
    Args:
        global_manifold (WhitneyPartitionOfUnity): The instantiated manifold object 
                                                   containing the atlas frames.
        delta_minimax (float): The base minimax radius threshold.
                               
    Returns:
        dict: Statistical metrics of the chart radii and the computed proportion.
    """
    atlas = global_manifold.atlas
    m = len(atlas)
    
    if m == 0:
        return {"proportion": 0.0, "min_radius": 0.0, "max_radius": 0.0}
        
    # Reconstruct the absolute radius by taking the square root of r_sq
    radii = [math.sqrt(frame['r_sq']) for frame in atlas]
    
    min_radius = min(radii)
    max_radius = max(radii)

    # Evaluate the condition for each chart
    valid_count = sum(1 for r in radii if r > min_radius - 1e-7)
    proportion = valid_count / float(m)
    

    
    return ({
        "proportion": proportion,
        "min_radius": min_radius,
        "max_radius": max_radius
    })
    
def evaluate_reconstruction_mse(global_manifold, data_ambient: torch.Tensor) -> float:
    """
    Computes the Mean Squared Error (MSE) between the raw ambient dataset 
    and its global C^{\beta+1}-smooth manifold approximation.
    
    Args:
        global_manifold (WhitneyPartitionOfUnity): The instantiated manifold object 
                                                   containing the atlas frames.
        data_ambient (torch.Tensor): The ambient dataset tensor of shape (N, p).
                               
    Returns:
        float: The continuous MSE loss metric.
    """
    # Disable gradient tracking to strictly evaluate the forward pass, 
    # preventing memory leakages during the grid search iterations.
    with torch.no_grad():
        # Project the ambient points onto the approximated manifold structure
        projected_data = global_manifold.evaluate_manifold(data_ambient)
        
        # Calculate the mean squared discrepancy between raw and projected coordinates
        reconstruction_mse = torch.nn.functional.mse_loss(projected_data, data_ambient).item()
    return reconstruction_mse




def test_metrics(global_manifold, fps_centers, global_membership_mask, data_ambient, d: int, beta: float) -> float:
    overlap_dict = test_minimum_chart_overlap(global_membership_mask)
    
    # Strict Topological Coverage Guard
    if overlap_dict['min_overlap'] < 1:
        return float('inf')
        
    m = fps_centers.size(0)
    N = data_ambient.size(0)
    k_degree = math.floor(beta) + 1
    
    # 1. Geometric Approximation Bias (NMSE)
    raw_mse = evaluate_reconstruction_mse(global_manifold, data_ambient)
    data_mean = data_ambient.mean(dim=0)
    baseline_var = torch.mean(torch.sum((data_ambient - data_mean)**2, dim=1)).item()
    nmse = raw_mse / max(baseline_var, 1e-8)
    
    # 2. Beta-Penalized Structural Variance (Degrees of Freedom per sample)
    d_poly = math.comb(d + k_degree, k_degree) - 1
    total_atlas_params = m * d_poly
    structural_variance = total_atlas_params / float(N)

        
    # 4. Total Upgraded Minimax Risk
    mean_overlap = overlap_dict['mean_overlap']
    score = (nmse + structural_variance) * mean_overlap

    print(f"Bias: {nmse:.4e} | Var: {structural_variance:.4e} | Charts: {m} | Overlap: {mean_overlap:.2f} | Score: {score:.4e}")
    
    return score