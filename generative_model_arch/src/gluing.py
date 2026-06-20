import torch

def compute_smooth_partition_of_unity(x: torch.Tensor, 
                                      centers_t: torch.Tensor, 
                                      precisions: torch.Tensor = None,
                                      length_scale: float = 1.0) -> torch.Tensor:
    batch_size, k_dim = x.shape
    m = centers_t.shape[0]
    
    if precisions is None:
        distances_sq = torch.cdist(x, centers_t) ** 2
    else:
        distances_sq = torch.zeros(batch_size, m, device=x.device)
        for i in range(m):
            delta = x - centers_t[i] 
            dist_i = torch.sum(torch.matmul(delta, precisions[i]) * delta, dim=1)
            distances_sq[:, i] = dist_i

    # Scale logits invariant to intrinsic dimension k to prevent softmax hard-saturation
    adjusted_scale = length_scale * torch.sqrt(torch.tensor(k_dim, dtype=torch.float32, device=x.device))
    logits = -distances_sq / (2 * adjusted_scale ** 2)
    
    return torch.softmax(logits, dim=1)

def compute_global_drift(x: torch.Tensor, 
                         feature_grads: torch.Tensor, 
                         local_etas: list, 
                         centers_t: torch.Tensor,
                         precisions: torch.Tensor = None) -> torch.Tensor:
    
    batch_size, k_dim = x.shape
    m = centers_t.shape[0]
    
    weights = compute_smooth_partition_of_unity(x, centers_t, precisions) 
    global_b_t = torch.zeros(batch_size, k_dim, device=x.device)
    
    for i in range(m):
        eta_i = local_etas[i]
        grads_transposed = feature_grads.transpose(1, 2)
        local_b_t = torch.matmul(grads_transposed, eta_i.unsqueeze(1)).squeeze(-1)
        
        rho_i = weights[:, i].unsqueeze(1) 
        global_b_t += rho_i * local_b_t
        
    return global_b_t