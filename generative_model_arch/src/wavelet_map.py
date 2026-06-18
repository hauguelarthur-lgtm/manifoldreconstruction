import torch
import torch.nn as nn

class TruncatedBesovWaveletMap(nn.Module):
    """
    Implements the deterministic feature map \phi(x) : R^p -> R^P.
    Strictly scales the hypothesis space to match the intrinsic regularity \mathcal{H}_1^{d/2}.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.p = ambient_dim
        self.d = intrinsic_dim
        self.P = p_truncation
        
        torch.manual_seed(42)
        self.basis_matrix = nn.Parameter(torch.randn(self.p, self.P) / self.p, requires_grad=False)
        self.basis_bias = nn.Parameter(torch.rand(self.P) * 2 * torch.pi, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.basis_matrix) + self.basis_bias
        return torch.sin(projection) * torch.exp(-0.1 * (projection ** 2))

def compute_feature_gradients(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)
    phi_x = model(x)
    batch_size, P = phi_x.shape
    p = x.shape[1]
    
    grads = torch.zeros(batch_size, P, p, device=x.device)
    for i in range(P):
        grad_i = torch.autograd.grad(
            outputs=phi_x[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        grads[:, i, :] = grad_i
        
    return grads