import torch
from src.wavelet_map import compute_feature_gradients
from src.diffusion import compute_optimal_diffusion

def compute_drift(X: torch.Tensor, model: torch.nn.Module, etas_t: list, chart_assignments: torch.Tensor, num_charts: int) -> torch.Tensor:
    """Computes the drift strictly on necessary particles to optimize memory and FLOPS."""
    b_t = torch.zeros_like(X)
    for i in range(num_charts):
        mask = (chart_assignments == i)
        if not mask.any(): continue
            
        X_i = X[mask]
        eta_i = etas_t[i].to(X.device)
        
        # Compute gradients exclusively for the sub-batch
        grads_i = compute_feature_gradients(model, X_i) 
        b_t[mask] = torch.matmul(grads_i.transpose(1, 2), eta_i.unsqueeze(1)).squeeze(-1)
        
    return b_t

def generate_samples(X_0: torch.Tensor,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     chart_assignments: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 100,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> torch.Tensor:
    
    X_t = X_0.clone().to(device)
    dt = 1.0 / num_time_steps
    time_grid = torch.linspace(0, 1.0 - dt, num_time_steps)
    model.eval()
    
    # Strict integration bound to prevent stiffness explosions
    max_drift = 15.0

    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        etas_t = precomputed_etas[step] 
        num_charts = len(etas_t)
        
        # Compute base forward drift
        b_t = compute_drift(X_t, model, etas_t, chart_assignments, num_charts)
        
        # Vector field clipping enforces a local Lipschitz bound on the discrete Euler step
        drift_norms = torch.norm(b_t, dim=1, keepdim=True)
        b_t = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
        
        if step % 10 == 0:
            print(f"Integration Step {step:02d} | t={t.item():.3f} | Mean L2 Drift: {torch.norm(b_t, dim=1).mean().item():.6f}")

        if ode_mode:
            # Execute Heun's Second-Order Method to reduce continuous-time truncation error
            X_pred = X_t + b_t * dt 
            
            if step < len(time_grid) - 1:
                etas_t_next = precomputed_etas[step + 1]
                b_t_next = compute_drift(X_pred, model, etas_t_next, chart_assignments, num_charts)
                
                drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                b_t_next = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                
                # Corrector step
                X_t = X_t + 0.5 * (b_t + b_t_next) * dt
            else:
                X_t = X_pred
        else:
            # Execute Euler-Maruyama SDE
            D_t_star = compute_optimal_diffusion(t.item())
            dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt, device=device))
            X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star, device=device)) * dW
            
    return X_t.detach()