import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.d = intrinsic_dim
        self.P = p_truncation
        self.omega = nn.Parameter(torch.zeros(self.P, self.d), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P) * 2 * np.pi, requires_grad=False)
        self.rkhs_penalty = nn.Parameter(torch.ones(self.P), requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def calibrate(self, x: torch.Tensor, base_beta: float = 1.5):
        with torch.no_grad():
            base_freq = float(np.clip(1.0 / max(torch.median(torch.pdist(x[:2000])).item(), 1e-4), 0.1, 10.0))
            dyadic_multipliers = [0.5, 1.0, 2.0, 4.0]
            p_per_dyad = self.P // len(dyadic_multipliers)
            
            omegas, penalties = [], []
            for mult in dyadic_multipliers:
                omegas.append(torch.randn(p_per_dyad, self.d) * (base_freq * mult))
                # Inject Besov \beta regularity into the RKHS norm diagonal
                penalties.append(torch.full((p_per_dyad,), float(mult ** (2.0 * base_beta))))
                
            self.omega.copy_(torch.cat(omegas, dim=0))
            self.rkhs_penalty.copy_(torch.cat(penalties, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(torch.matmul(x, self.omega.T) + self.bias)

    def get_rkhs_penalty(self) -> torch.Tensor:
        return self.rkhs_penalty