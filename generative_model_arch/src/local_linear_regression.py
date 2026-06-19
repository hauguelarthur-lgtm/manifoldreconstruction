import torch

def solve_local_system(feature_grads: torch.Tensor, target_vectors: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """
    Relaxed Ridge Regression: Allows the vector field to fully reach target coordinates
    without premature shrinking, while preventing numerical singularity.
    """
    N_i, P, p = feature_grads.shape
    A = feature_grads.transpose(1, 2).reshape(N_i * p, P)
    B = target_vectors.reshape(N_i * p, 1)
    
    reg_tensor = torch.sqrt(torch.tensor(reg, device=A.device))
    eye_P = torch.eye(P, device=A.device) * reg_tensor
    A_aug = torch.cat([A, eye_P], dim=0)
    
    zero_aug = torch.zeros(P, 1, device=B.device)
    B_aug = torch.cat([B, zero_aug], dim=0)
    
    eta_t = torch.linalg.lstsq(A_aug, B_aug, rcond=None).solution.squeeze(-1)
    return eta_t