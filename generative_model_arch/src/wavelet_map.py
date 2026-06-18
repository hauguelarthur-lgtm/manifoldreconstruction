import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    Implements Random Fourier Features (RFF).
    Guarantees non-vanishing gradients globally while retaining 
    high-frequency local expressiveness for the linear regression.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.p = ambient_dim
        self.d = intrinsic_dim
        self.P = p_truncation
        
        torch.manual_seed(42)
        # Frequency matrix \Omega and phase shift b
        # Scaled by 1/sqrt(p) to maintain stable dot products in high dimensions
        self.omega = nn.Parameter(torch.randn(self.P, self.p) / np.sqrt(self.p), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P) * 2 * np.pi, requires_grad=False)
        
        # RFF normalization constant
        self.scale = float(np.sqrt(2.0 / self.P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection shape: (batch_size, P)
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)

def compute_feature_gradients(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Computes exact analytical gradients of the RFF map.
    Bypasses PyTorch autograd for computational stability and speed.
    """
    # projection shape: (batch_size, P)
    projection = torch.matmul(x, model.omega.T) + model.bias
    
    # Derivative of cos is -sin
    # S shape: (batch_size, P)
    S = -model.scale * torch.sin(projection)
    
    # Analytical gradient: \nabla \phi_k(x) = S_k * \Omega_k
    # S.unsqueeze(-1) shape: (batch_size, P, 1)
    # model.omega.unsqueeze(0) shape: (1, P, p)
    # Resulting grads shape: (batch_size, P, p)
    grads = S.unsqueeze(-1) * model.omega.unsqueeze(0)
    
    return grads