import torch

def solve_local_system(feature_grads: torch.Tensor, target_vectors: torch.Tensor, reg: float = 1e-2) -> torch.Tensor:
    """
    Solves local regression using an Augmented Matrix formulation to strictly 
    bound ||\eta||_2 and prevent numerical scaling explosions.
    """
    N_i, P, p = feature_grads.shape
    phi_matrix = feature_grads.transpose(1, 2).reshape(N_i * p, P)
    target = target_vectors.reshape(N_i * p, 1)
    
    # Construct augmented matrices for exact Ridge Regression
    reg_tensor = torch.sqrt(torch.tensor(reg, device=phi_matrix.device))
    eye_P = torch.eye(P, device=phi_matrix.device) * reg_tensor
    phi_aug = torch.cat([phi_matrix, eye_P], dim=0)
    
    zero_aug = torch.zeros(P, 1, device=target.device)
    target_aug = torch.cat([target, zero_aug], dim=0)
    
    eta_t = torch.linalg.lstsq(phi_aug, target_aug, rcond=None).solution.squeeze(-1)
    
    return eta_t