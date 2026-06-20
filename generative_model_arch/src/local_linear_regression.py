import torch

def solve_local_system(feature_grads: torch.Tensor, target_vectors: torch.Tensor, reg: float = 1e-2) -> torch.Tensor:
    """
    Solves local regression using the Normal Equations for CPU acceleration.
    """
    N_i, P, p = feature_grads.shape
    phi_matrix = feature_grads.transpose(1, 2).reshape(N_i * p, P)
    target = target_vectors.reshape(N_i * p, 1)
    
    # Compute the empirical Gram matrix (P x P)
    # This matrix multiplication is highly optimized on CPUs
    gram = torch.matmul(phi_matrix.t(), phi_matrix)
    
    # Add Tikhonov regularization
    gram += torch.eye(P, device=phi_matrix.device) * reg
    
    # Compute the right-hand side vector (P x 1)
    rhs = torch.matmul(phi_matrix.t(), target)
    
    # Solve the positive-definite system via Cholesky decomposition
    # Much faster than SVD/QR on the full N*p x P matrix
    eta_t = torch.linalg.solve(gram, rhs).squeeze(-1)
    
    return eta_t