import torch

def solve_local_system(features: torch.Tensor, 
                       target_velocities: torch.Tensor, 
                       reg: float = 1e-2) -> torch.Tensor:
    """
    Solves Kernel Ridge Regression for general vector fields (arXiv:2602.20070).
    System: \Phi^\top \Phi \eta = \Phi^\top Y  --> Solves for \eta \in R^{P \times d}.
    Stripped of all invalid velocity normalization concepts and memory allocation churn.
    """
    P = features.shape[1]
    
    # Gram matrix G \in R^{P \times P}
    gram = torch.matmul(features.t(), features)
    
    # Scale-invariant trace regularization executed in-place without tensor allocation
    trace_scale = torch.trace(gram) / P
    gram.diagonal().add_(reg * trace_scale + 1e-6)
    
    # RHS \in R^{P \times d} executed against true, un-truncated physical velocities
    rhs = torch.matmul(features.t(), target_velocities)
    
    # Solves exactly for coefficient matrix \eta \in R^{P \times d}
    eta_t = torch.linalg.solve(gram, rhs)
    
    return eta_t