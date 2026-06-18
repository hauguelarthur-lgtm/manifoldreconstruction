import torch

def compute_smooth_partition_of_unity(x: torch.Tensor, centers: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    """
    Computes \rho_i(x) \in \mathcal{H}_C^{\beta+1}.
    Ensures \sum \rho_i(x) = 1 across all charts.
    """
    distances = torch.cdist(x, centers) 
    logits = - (distances ** 2) / (2 * length_scale ** 2)
    weights = torch.softmax(logits, dim=1)
    
    return weights

def compute_global_drift(x: torch.Tensor, 
                         feature_grads: torch.Tensor, 
                         local_etas: list, 
                         centers: torch.Tensor) -> torch.Tensor:
    """
    Executes the convex combination of local velocity fields.
    Equation: \hat{b}_t(x) = \sum_{i=1}^m \rho_i(x) ( \nabla\phi_i(x)^\top \eta_t^{(i)} )
    """
    batch_size, p = x.shape
    m = centers.shape[0]
    
    weights = compute_smooth_partition_of_unity(x, centers) 
    global_b_t = torch.zeros(batch_size, p, device=x.device)
    
    for i in range(m):
        eta_i = local_etas[i]
        grads_transposed = feature_grads.transpose(1, 2)
        local_b_t = torch.matmul(grads_transposed, eta_i.unsqueeze(1)).squeeze(-1)
        
        rho_i = weights[:, i].unsqueeze(1) 
        global_b_t += rho_i * local_b_t
        
    return global_b_t