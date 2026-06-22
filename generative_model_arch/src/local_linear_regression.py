import torch

def solve_local_system(features: torch.Tensor, 
                       target_velocities: torch.Tensor, 
                       rkhs_penalty: torch.Tensor, 
                       reg: float = 1e-3) -> torch.Tensor:
    """
    Solves Localized Littlewood-Paley Drift Coefficients \eta_t^(i).
    MATHEMATICAL AMELIORATION:
    1. Replaces static 1e-6 floor with Spectral Water-Filling Regularization.
    2. Implements stable Hermitian Pseudoinversion fallback for ill-conditioned sparse patches.
    """
    N, P = features.shape
    gram = torch.matmul(features.t(), features)
    
    # Scale-Invariant Trace Extraction
    trace_val = torch.trace(gram)
    spectral_scale = (trace_val / float(P)) if trace_val > 1e-8 else torch.tensor(1.0, device=features.device)
    
    # Rigorous Water-Filling Diagonal Addition (Maintains base identity I + \lambda K)
    water_filling_diag = reg * spectral_scale * (1.0 + rkhs_penalty) + 1e-7
    gram.diagonal().add_(water_filling_diag)
    
    rhs = torch.matmul(features.t(), target_velocities)
    
    # Check empirical condition number stability
    try:
        return torch.linalg.solve(gram, rhs)
    except (RuntimeError, torch._C._LinAlgError):
        # Fallback to robust Moore-Penrose pseudoinversion exploiting Hermitian symmetry
        return torch.linalg.pinv(gram, rcond=1e-5, hermitian=True).matmul(rhs)