import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    True Besov Wavelet Map (arXiv:2506.19587).
    Uses Multiscale Dyadic Cauchy sampling to generate heavy-tailed frequencies,
    spanning adaptive Besov space B_{p,q}^\beta inside intrinsic dimension 'd'.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.d = intrinsic_dim
        self.P = p_truncation
        
        gen = torch.Generator()
        gen.manual_seed(42)
        
        self.omega = nn.Parameter(torch.zeros(self.P, self.d), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P, generator=gen) * 2 * np.pi, requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def calibrate(self, x: torch.Tensor, base_beta: float = 1.5):
        """Calibrates heavy-tailed dyadic Cauchy scales to match Besov regularity \beta."""
        with torch.no_grad():
            distances = torch.pdist(x[:2000], p=2)
            median_dist = torch.median(distances).item()
            
            # Structurally clamp base frequency between [0.1, 10.0] to guarantee gradient stability
            base_freq = float(np.clip(1.0 / max(median_dist, 1e-4), 0.1, 10.0))
            
            dyadic_multipliers = [0.5, 1.0, 2.0, 4.0]
            p_per_dyad = self.P // len(dyadic_multipliers)
            
            omegas = []
            for mult in dyadic_multipliers:
                scale_j = base_freq * mult
                cauchy_weights = torch.randn(p_per_dyad, self.d, device=x.device) / \
                                (torch.randn(p_per_dyad, 1, device=x.device).abs() + 1e-5)
                
                weights_j = cauchy_weights * (scale_j * (mult ** (-base_beta)))
                omegas.append(weights_j)
                
            if self.P % len(dyadic_multipliers) != 0:
                rem = self.P % len(dyadic_multipliers)
                omegas.append(torch.randn(rem, self.d, device=x.device) * base_freq)
                
            self.omega.copy_(torch.cat(omegas, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)