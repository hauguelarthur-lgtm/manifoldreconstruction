import torch

def solve_local_system(feature_grads: torch.Tensor, target_vectors: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """
    Constructs the empirical Gram matrix and solves the local P x P linear regression.
    Equation: \hat{K}_t^{(i)} \eta_t^{(i)} = \hat{r}_t^{(i)}
    """
    N_i, P, p = feature_grads.shape
    
    K_t = torch.zeros(P, P, device=feature_grads.device)
    for i in range(N_i):
        grad_matrix = feature_grads[i] 
        K_t += torch.matmul(grad_matrix, grad_matrix.T)
    K_t /= N_i
    
    K_t += reg * torch.eye(P, device=feature_grads.device)
    
    r_t = torch.zeros(P, device=feature_grads.device)
    for i in range(N_i):
        grad_matrix = feature_grads[i] 
        target = target_vectors[i]     
        r_t += torch.matmul(grad_matrix, target)
    r_t /= N_i
    
    eta_t = torch.linalg.solve(K_t, r_t)
    #Test coefficient collapse hypothesis
    #eta_t = torch.linalg.lstsq(K_t, r_t, rcond=1e-5).solution
    
    return eta_t

