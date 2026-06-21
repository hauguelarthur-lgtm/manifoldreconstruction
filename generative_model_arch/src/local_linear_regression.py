import torch

def solve_local_system(features: torch.Tensor, 
                       target_velocities: torch.Tensor, 
                       reg: float = 1e-2) -> torch.Tensor:
    """
    Solves Kernel Ridge Regression for general, non-irrotational vector fields (arXiv:2602.20070).
    System: \Phi^\top \Phi \eta = \Phi^\top Y  --> Solves for \eta \in R^{P \times d}.
    """
    P = features.shape[1]
    
    # Gram matrix G \in R^{P \times P}
    gram = torch.matmul(features.t(), features)
    
    trace_scale = torch.trace(gram) / P
    gram += torch.eye(P, device=features.device) * (reg * trace_scale + 1e-6)
    
    # RHS \in R^{P \times d}
    rhs = torch.matmul(features.t(), target_velocities)
    
    # Solves exactly for coefficient matrix \eta \in R^{P \times d}
    # Unlocks full non-zero curl (\nabla \times b \neq 0)
    eta_t = torch.linalg.solve(gram, rhs)
    
    return eta_t