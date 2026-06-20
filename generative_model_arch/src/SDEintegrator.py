import torch
from src.wavelet_map import compute_feature_gradients
from src.diffusion import compute_optimal_diffusion
from src.gluing import compute_global_drift

def generate_samples(X_0: torch.Tensor,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     cluster_centers: torch.Tensor,
                     cluster_precisions: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 500,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> torch.Tensor:
    
    X_t = X_0.clone().to(device)
    cluster_centers = cluster_centers.to(device)
    model.eval()
    
    # Enforce strict uniform temporal synchronization
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    time_grid = torch.linspace(0, 1.0 - epsilon_num, num_time_steps)
    max_drift = 15.0

    # Pre-allocate static scalars to prevent synchronous CPU/GPU blocking
    dt_tensor = torch.tensor(dt, device=device)

    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        etas_t = precomputed_etas[step] 
        t_val = t.item()
        
        # Predictor Drift Computation (Probability Flow Velocity)
        feature_grads = compute_feature_gradients(model, X_t)
        b_t = compute_global_drift(X_t, feature_grads, etas_t, cluster_centers, precisions=cluster_precisions)
        
        drift_norms = torch.norm(b_t, dim=1, keepdim=True)
        b_t = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)

        if ode_mode:
            # Heun's Method (2nd Order Deterministic Probability Flow)
            X_pred = X_t + b_t * dt 
            if step < len(time_grid) - 1:
                etas_t_next = precomputed_etas[step + 1]
                feature_grads_next = compute_feature_gradients(model, X_pred)
                b_t_next = compute_global_drift(X_pred, feature_grads_next, etas_t_next, cluster_centers, precisions=cluster_precisions)
                
                drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                b_t_next = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                
                X_t = X_t + 0.5 * (b_t + b_t_next) * dt
            else:
                X_t = X_pred
        else:
            # Euler-Maruyama Method (Stochastic Interpolant Framework)
            D_t_star = compute_optimal_diffusion(t_val)
            D_t_tensor = torch.tensor(D_t_star, device=device)
            
            # 1. Algebraic Score Mapping 
            # Derivation: s_t = (beta_t * b_t(x) - x) / (alpha_t * gamma_t)
            # Evaluated for the linear interpolant schedule.
            # Max clamp applied to the denominator to prevent division by zero strictly at t=1 boundary.
            s_t = (t_val * b_t - X_t) / max(1.0 - t_val, 1e-8)
            
            # 2. Score-Corrected Drift formulation
            # Integrates the marginal score function scaled by the diffusion coefficient.
            full_drift = b_t + D_t_tensor * s_t
            
            # 3. Discrete Stochastic Integration Step
            # The seamless analytical formulation natively handles the t=0 boundary.
            dW = torch.randn_like(X_t) * torch.sqrt(dt_tensor)
            X_t = X_t + full_drift * dt + torch.sqrt(2 * D_t_tensor) * dW

    # Terminal Boundary Projection (Epsilon gap closure)
    etas_final = precomputed_etas[-1]
    feature_grads_final = compute_feature_gradients(model, X_t)
    b_t_final = compute_global_drift(X_t, feature_grads_final, etas_final, cluster_centers, precisions=cluster_precisions)
    
    X_t = X_t + b_t_final * epsilon_num
        
    return X_t.detach()