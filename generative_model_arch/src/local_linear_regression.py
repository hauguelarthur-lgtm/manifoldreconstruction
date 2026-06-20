import torch

def solve_local_system(feature_grads: torch.Tensor, target_vectors: torch.Tensor, reg: float = 1e-2) -> torch.Tensor:
    """
    Solves local regression utilizing Cholesky factorization on the Tikhonov-regularized 
    Gram matrix to guarantee positive-definite structural stability.
    """
    N_i, P, p = feature_grads.shape
    
    # Reshape tensors for optimal spatial evaluation
    phi_matrix = feature_grads.transpose(1, 2).reshape(N_i * p, P)
    target = target_vectors.reshape(N_i * p, 1)
    
    # Compute the empirical Gram matrix (P x P)
    gram = torch.matmul(phi_matrix.t(), phi_matrix)
    
    # Inject strict Tikhonov regularization to enforce positive-definiteness
    gram += torch.eye(P, device=phi_matrix.device) * reg
    
    # Compute the right-hand side projected vector
    rhs = torch.matmul(phi_matrix.t(), target)
    
    try:
        # Execute exact Cholesky decomposition (L * L^T = Gram)
        L = torch.linalg.cholesky(gram)
        # Solve the factorized system strictly algebraically
        eta_t = torch.cholesky_solve(rhs, L).squeeze(-1)
        
    except torch.linalg.LinAlgError:
        # Algorithmic fallback: If numerical floating-point limitations are breached 
        # (e.g. collinear feature responses dominating the regularization scalar),
        # compute the Moore-Penrose pseudo-inverse via SVD.
        # rcond=1e-5 explicitly truncates the degenerate zero-subspace eigenvalues.
        eta_t = torch.matmul(torch.linalg.pinv(gram, rcond=1e-5), rhs).squeeze(-1)
        
    return eta_t