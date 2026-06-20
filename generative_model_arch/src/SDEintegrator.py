import torch
from src.wavelet_map import compute_feature_gradients
from src.diffusion import compute_optimal_diffusion
from src.gluing import compute_global_drift

def generate_samples(X_0: torch.Tensor,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     cluster_centers: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 100,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False,
                     epsilon: float = 1e-3) -> torch.Tensor:
    
    X_t = X_0.clone().to(device)
    cluster_centers = cluster_centers.to(device)
    model.eval()
    
    # Epsilon-Truncation Grid
    truncated_steps = int(num_time_steps * (1.0 - epsilon))
    dt = (1.0 - epsilon) / truncated_steps
    time_grid = torch.linspace(0, 1.0 - epsilon - dt, truncated_steps)
    max_drift = 15.0

    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        etas_t = precomputed_etas[step] 
        
        # 1. Compute global feature gradients dynamically for the current spatial locations
        feature_grads = compute_feature_gradients(model, X_t)
        
        # 2. Execute smooth topological gluing based on real-time Euclidean barycenters
        # This replaces the invalid static boolean mask
        b_t = compute_global_drift(X_t, feature_grads, etas_t, cluster_centers)
        
        drift_norms = torch.norm(b_t, dim=1, keepdim=True)
        b_t = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
        
        if step % 10 == 0:
            print(f"Integration Step {step:02d} | t={t.item():.3f} | Mean L2 Drift: {torch.norm(b_t, dim=1).mean().item():.6f}")

        if ode_mode:
            X_pred = X_t + b_t * dt 
            if step < len(time_grid) - 1:
                etas_t_next = precomputed_etas[step + 1]
                feature_grads_next = compute_feature_gradients(model, X_pred)
                b_t_next = compute_global_drift(X_pred, feature_grads_next, etas_t_next, cluster_centers)
                
                drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                b_t_next = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                
                X_t = X_t + 0.5 * (b_t + b_t_next) * dt
            else:
                X_t = X_pred
        else:
            if step == 0:
                X_t = X_t + b_t * dt
            else:
                D_t_star = compute_optimal_diffusion(t.item())
                dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt, device=device))
                X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star, device=device)) * dW
                
    # Terminal Projection Phase
    etas_final = precomputed_etas[-1]
    feature_grads_final = compute_feature_gradients(model, X_t)
    b_t_final = compute_global_drift(X_t, feature_grads_final, etas_final, cluster_centers)
    
    X_t = X_t + b_t_final * epsilon
        
    return X_t.detach()