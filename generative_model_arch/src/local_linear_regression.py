import torch

def solve_local_system(features: torch.Tensor, target_velocities: torch.Tensor, rkhs_penalty: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    P = features.shape[1]
    gram = torch.matmul(features.t(), features)
    # Inject Besov RKHS norm penalty directly into the Gram diagonal
    gram.diagonal().add_(reg * (torch.trace(gram)/P) * (1.0 + rkhs_penalty) + 1e-6)
    return torch.linalg.solve(gram, torch.matmul(features.t(), target_velocities))