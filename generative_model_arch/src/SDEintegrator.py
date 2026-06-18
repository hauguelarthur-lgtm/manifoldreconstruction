import torch
from src.wavelet_map import compute_feature_gradients
from src.gluing import compute_global_drift
from src.diffusion import compute_optimal_diffusion

def generate_samples(model: torch.nn.Module,
                     precomputed_etas: list, 
                     cluster_centers: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 100,
                     device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Executes the generative stochastic differential equation (SDE) using the Euler-Maruyama scheme.
    Equation: dX_t = \hat{b}_t(X_t)dt + \sqrt{2 D_t^*} dW_t
    """
    X_t = torch.randn(num_samples, ambient_dim, device=device)
    dt = 1.0 / num_time_steps
    time_grid = torch.linspace(0, 1.0 - dt, num_time_steps)
    
    model.eval()
    
    for step, t in enumerate(time_grid):
        feature_grads = compute_feature_gradients(model, X_t)
        etas_t = precomputed_etas[step]
        
        b_t = compute_global_drift(X_t, feature_grads, etas_t, cluster_centers)
        D_t_star = compute_optimal_diffusion(t.item())
        
        dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt))
        X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star)) * dW
        
    return X_t.detach()