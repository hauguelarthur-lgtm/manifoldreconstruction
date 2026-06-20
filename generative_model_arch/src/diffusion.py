import torch

def compute_optimal_diffusion(t: float) -> float:
    """
    Computes the exact optimal Girsanov diffusion coefficient.
    Analytically reduced for the linear interpolant schedule to resolve the t=0 singularity.
    """
    return (1.0 - t) ** 2