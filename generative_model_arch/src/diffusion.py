import torch

def compute_optimal_diffusion(t: float, t_eps: float = 1e-4) -> float:
    """
    Computes the optimal diffusion coefficient strictly bounded 
    by the parabolic noise schedule gamma_t = t * (1 - t).
    """
    t = max(t, t_eps)
    alpha_t = 1.0 - t
    beta_t = t
    
    # Bounded interpolant variance bridge
    gamma_t = t * (1.0 - t)
    
    D_t_star = (alpha_t * gamma_t) / beta_t
    return D_t_star