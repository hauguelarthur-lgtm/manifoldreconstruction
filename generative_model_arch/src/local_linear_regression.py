import torch

def solve_local_system(feature_grads: torch.Tensor, 
                       target_vectors: torch.Tensor, 
                       reg: float = 1e-2) -> torch.Tensor:
    """
    Solves the batched linear least-squares system for intrinsic KSI drift regression.
    Models the probability flow velocity field as a conservative potential flow inside R^d.
    """
    N_i, P, d = feature_grads.shape
    
    # Enforce strict memory contiguity prior to reshaping to guarantee correct coordinate alignment
    phi_matrix = feature_grads.transpose(1, 2).contiguous().reshape(N_i * d, P)
    target = target_vectors.contiguous().reshape(N_i * d, 1)
    
    # Formulate empirical Gram matrix (P x P)
    gram = torch.matmul(phi_matrix.t(), phi_matrix)
    
    # Invariant Tikhonov Regularization scaled by Gram trace
    trace_scale = torch.trace(gram) / P
    gram += torch.eye(P, device=phi_matrix.device) * (reg * trace_scale + 1e-6)
    
    rhs = torch.matmul(phi_matrix.t(), target)
    
    # Utilize high-precision linear solve instead of explicit pseudo-inverse forming
    eta_t = torch.linalg.solve(gram, rhs).squeeze(-1)
    
    return eta_t