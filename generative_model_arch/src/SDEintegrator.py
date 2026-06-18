import torch
from src.wavelet_map import compute_feature_gradients
from src.diffusion import compute_optimal_diffusion

def generate_samples(model: torch.nn.Module,
                     precomputed_etas: list, 
                     chart_assignments: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 100,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> torch.Tensor:
    """
    Executes integration using a Mixture of Independent Flows.
    Bypasses spatial vector field gluing to eliminate destructive interference at the origin.
    """
    X_t = torch.randn(num_samples, ambient_dim, device=device)
    dt = 1.0 / num_time_steps
    time_grid = torch.linspace(0, 1.0 - dt, num_time_steps)
    
    model.eval()
    
    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        
        feature_grads = compute_feature_gradients(model, X_t) # Shape: (N, P, p)
        etas_t = precomputed_etas[step] 
        num_charts = len(etas_t)
        
        # Dynamically construct the vector field strictly respecting chart assignments
        b_t = torch.zeros_like(X_t)
        
        for i in range(num_charts):
            mask = (chart_assignments == i)
            if not mask.any():
                continue
                
            eta_i = etas_t[i].to(device)
            grads_i = feature_grads[mask] # (N_i, P, p)
            
            # Execute batch matrix multiplication for this specific chart
            b_t[mask] = torch.matmul(grads_i.transpose(1, 2), eta_i.unsqueeze(1)).squeeze(-1)
        
        if step % 10 == 0:
            drift_magnitude = torch.norm(b_t, dim=1).mean().item()
            print(f"Integration Step {step:02d} | t={t.item():.3f} | Mean Drift L2 Norm: {drift_magnitude:.6f}")
        
        if ode_mode:
            X_t = X_t + b_t * dt
        else:
            D_t_star = compute_optimal_diffusion(t.item())
            dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt, device=device))
            X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star, device=device)) * dW
            
    return X_t.detach()