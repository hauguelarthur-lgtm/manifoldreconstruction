import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    True Besov Wavelet Map (arXiv:2506.19587).
    EXACT CORRECTION: Bakes dyadic Besov regularity \beta directly into the RKHS 
    Sobolev penalty diagonal while enforcing hard Cauchy Nyquist clamps.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.d = intrinsic_dim
        self.P = p_truncation
        
        gen = torch.Generator()
        gen.manual_seed(42)
        
        self.omega = nn.Parameter(torch.zeros(self.P, self.d), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P, generator=gen) * 2 * np.pi, requires_grad=False)
        self.rkhs_penalty = nn.Parameter(torch.ones(self.P), requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def calibrate(self, x: torch.Tensor, base_beta: float = 1.5):
        with torch.no_grad():
            distances = torch.pdist(x[:2000], p=2)
            base_freq = float(np.clip(1.0 / max(torch.median(distances).item(), 1e-4), 0.1, 10.0))
            
            dyadic_multipliers = [0.5, 1.0, 2.0, 4.0]
            p_per_dyad = self.P // len(dyadic_multipliers)
            
            omegas = []
            penalties = []
            for mult in dyadic_multipliers:
                cauchy_raw = torch.randn(p_per_dyad, self.d, device=x.device) / \
                            (torch.randn(p_per_dyad, 1, device=x.device).abs() + 1e-5)
                
                # MATHEMATICAL FIX 3: Cauchy Nyquist Aliasing Clamp
                cauchy_clamped = torch.clamp(cauchy_raw, min=-50.0, max=50.0)
                omegas.append(cauchy_clamped * (base_freq * mult))
                
                # Sobolev RKHS norm penalty scales strictly upwards: (2^j)^(2\beta)
                penalties.append(torch.full((p_per_dyad,), float(mult ** (2.0 * base_beta)), device=x.device))
                
            if self.P % len(dyadic_multipliers) != 0:
                rem = self.P % len(dyadic_multipliers)
                omegas.append(torch.randn(rem, self.d, device=x.device) * base_freq)
                penalties.append(torch.ones(rem, device=x.device))
                
            self.omega.copy_(torch.cat(omegas, dim=0))
            self.rkhs_penalty.copy_(torch.cat(penalties, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)

    def get_rkhs_penalty(self) -> torch.Tensor:
        return self.rkhs_penalty