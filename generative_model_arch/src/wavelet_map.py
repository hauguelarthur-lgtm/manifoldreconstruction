import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    Multi-Scale Random Fourier Features.
    Divides the P basis functions across multiple frequency bandwidths 
    to capture both global topology and local high-frequency curvature.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.p = ambient_dim
        self.d = intrinsic_dim
        self.P = p_truncation
        
        torch.manual_seed(42)
        
        # Define 3 scales: Coarse, Medium, Fine
        scales = [0.5, 1.0, 2.0]
        p_per_scale = self.P // len(scales)
        
        omegas = []
        for scale in scales:
            # Scale frequencies by 1/sqrt(p) * scale
            w = torch.randn(p_per_scale, self.p) * (scale / np.sqrt(self.p))
            omegas.append(w)
            
        # Handle remainder if P is not divisible by len(scales)
        if self.P % len(scales) != 0:
            rem = self.P % len(scales)
            omegas.append(torch.randn(rem, self.p) * (1.0 / np.sqrt(self.p)))
            
        self.omega = nn.Parameter(torch.cat(omegas, dim=0), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P) * 2 * np.pi, requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)

def compute_feature_gradients(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    projection = torch.matmul(x, model.omega.T) + model.bias
    S = -model.scale * torch.sin(projection)
    grads = S.unsqueeze(-1) * model.omega.unsqueeze(0)
    return grads