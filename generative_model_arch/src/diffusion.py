import torch

def compute_optimal_diffusion(t: float, t_eps: float = 1e-4) -> float:
    """
    Computes the optimal diffusion coefficient D_t^* = (alpha_t * gamma_t) / beta_t
    derived from Girsanov's theorem to strictly bound estimation errors.
    """
    t = max(t, t_eps)
    alpha_t = 1.0 - t
    beta_t = t
    gamma_t = 1.0 
    
    D_t_star = (alpha_t * gamma_t) / beta_t
    return D_t_star