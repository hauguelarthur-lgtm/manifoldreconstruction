import torch
from src.wavelet_map import compute_feature_gradients
from src.diffusion import compute_optimal_diffusion
from src.gluing import compute_global_drift

def generate_samples(X_0: torch.Tensor,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     cluster_centers: torch.Tensor,
                     cluster_precisions:torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 500,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False,
                     corrector_steps: int = 1,
                     snr: float = 0.15) -> torch.Tensor:
    
    X_t = X_0.clone().to(device)
    cluster_centers = cluster_centers.to(device)
    model.eval()
    
    epsilon_num = 1e-3
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    time_grid = torch.linspace(0, 1.0 - 1e-5, num_time_steps)
    max_drift = 15.0

    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        etas_t = precomputed_etas[step] 
        
        # Predictor Drift Computation
        feature_grads = compute_feature_gradients(model, X_t)
        b_t = compute_global_drift(X_t, feature_grads, etas_t, cluster_centers, precisions=cluster_precisions)
        
        drift_norms = torch.norm(b_t, dim=1, keepdim=True)
        b_t = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)

        if ode_mode:
            # Heun's Method
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
            # Predictor: Euler-Maruyama
            if step == 0:
                X_t = X_t + b_t * dt
            else:
                D_t_star = compute_optimal_diffusion(t.item())
                dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt, device=device))
                X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star, device=device)) * dW
                
                # Corrector: Annealed Langevin Dynamics
                # Executes exclusively in the stochastic domain to contract orthogonal variance
                for _ in range(corrector_steps):
                    # Recompute gradient at the stochastically perturbed state
                    grad_corr = compute_feature_gradients(model, X_t)
                    b_corr = compute_global_drift(X_t, grad_corr, etas_t, cluster_centers, precisions=cluster_precisions)
                    
                    # Compute signal-to-noise ratio-based step size
                    noise_norm = torch.randn_like(X_t)
                    grad_norm = torch.norm(b_corr, dim=1).mean()
                    noise_mean_norm = torch.norm(noise_norm, dim=1).mean()
                    
                    step_size = (snr * noise_mean_norm / (grad_norm + 1e-8)) ** 2 * 2 * D_t_star
                    
                    # Execute Langevin step to project onto intrinsic support
                    X_t = X_t + step_size * b_corr + torch.sqrt(2 * step_size) * noise_norm
                
    return X_t.detach()