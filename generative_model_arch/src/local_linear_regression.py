import torch

def solve_local_system(features: torch.Tensor, 
                       target_velocities: torch.Tensor, 
                       rkhs_penalty: torch.Tensor,
                       reg: float = 1e-3) -> torch.Tensor:
    """
    Solves Kernel Ridge Regression for general vector fields (arXiv:2602.20070).
    EXACT CORRECTION: Injects the Besov RKHS norm penalty directly into the Gram diagonal.
    """
    P = features.shape[1]
    gram = torch.matmul(features.t(), features)
    
    trace_scale = torch.trace(gram) / P
    
    # In-place diagonal addition executed without GPU memory allocation
    gram.diagonal().add_(rkhs_penalty * (reg * trace_scale) + 1e-6)
    
    rhs = torch.matmul(features.t(), target_velocities)
    return torch.linalg.solve(gram, rhs)