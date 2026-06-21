import torch

def solve_local_system(features: torch.Tensor, 
                       target_velocities: torch.Tensor, 
                       rkhs_penalty: torch.Tensor,
                       reg: float = 1e-3) -> torch.Tensor:
    P = features.shape[1]
    gram = torch.matmul(features.t(), features)
    
    trace_scale = torch.trace(gram) / P
    
    # MATHEMATICAL FIX 4: Baseline Ridge Conditioning Floor
    # Guarantees full matrix rank independent of small dyadic penalty blocks.
    gram.diagonal().add_(reg * trace_scale * (1.0 + rkhs_penalty) + 1e-6)
    
    rhs = torch.matmul(features.t(), target_velocities)
    return torch.linalg.solve(gram, rhs)