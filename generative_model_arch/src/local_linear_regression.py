import torch

def solve_local_system(feature_grads: torch.Tensor, 
                       target_vectors: torch.Tensor, 
                       reg: float = 1e-2) -> torch.Tensor:
    """
    Solves the batched linear system for the probability flow vector field.
    """
    N_i, P, k = feature_grads.shape
    
    phi_matrix = feature_grads.transpose(1, 2).reshape(N_i * k, P)
    target = target_vectors.reshape(N_i * k, 1)
    
    gram = torch.matmul(phi_matrix.t(), phi_matrix)
    
    # Invariant Tikhonov Regularization
    # Scale penalty by the spatial trace to prevent under-regularization 
    # in dense charts and over-regularization in sparse charts.
    trace_scale = torch.trace(gram) / P
    gram += torch.eye(P, device=phi_matrix.device) * (reg * trace_scale + 1e-6)
    
    rhs = torch.matmul(phi_matrix.t(), target)
    
    eta_t = torch.matmul(torch.linalg.pinv(gram, rcond=1e-5), rhs).squeeze(-1)
    
    return eta_t