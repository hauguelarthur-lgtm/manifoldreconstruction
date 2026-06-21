import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    Multi-Scale Random Fourier Features (Truncated Besov Wavelet Map).
    Parameterizes the scaling space across multiple frequency bandwidths 
    natively inside the intrinsic dimension 'd'.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        # For Phase 3 intrinsic regression, ambient_dim == intrinsic_dim == d
        self.p = ambient_dim      
        self.d = intrinsic_dim
        self.P = p_truncation
        
        # Isolate random generator state to prevent global seed collapse
        gen = torch.Generator()
        gen.manual_seed(42)
        
        self.omega = nn.Parameter(torch.randn(self.P, self.p, generator=gen), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P, generator=gen) * 2 * np.pi, requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def calibrate(self, x: torch.Tensor, subsample: int = 2000):
        """
        Dynamically calibrates frequency bandwidths using the median pairwise 
        distance heuristic of the empirical intrinsic coordinates.
        """
        with torch.no_grad():
            N = x.size(0)
            if N > subsample:
                indices = torch.randperm(N, device=x.device)[:subsample]
                x_sub = x[indices]
            else:
                x_sub = x
            
            distances = torch.pdist(x_sub, p=2)
            
            median_dist = torch.median(distances).item()
            if median_dist <= 1e-8:
                median_dist = 1.0  
            
            base_freq = 1.0 / median_dist
            
            # Distribute basis functions across dyadic frequency scales
            scales = [base_freq * 0.5, base_freq, base_freq * 2.0]
            p_per_scale = self.P // len(scales)
            
            omegas = []
            for scale in scales:
                w = torch.randn(p_per_scale, self.p, device=x.device) * (scale / np.sqrt(self.p))
                omegas.append(w)
                
            if self.P % len(scales) != 0:
                rem = self.P % len(scales)
                omegas.append(torch.randn(rem, self.p, device=x.device) * (base_freq / np.sqrt(self.p)))
                
            self.omega.copy_(torch.cat(omegas, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)

def compute_feature_gradients(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Computes exact analytical Jacobian feature gradients \nabla\phi(x).
    Returns tensor of shape (Batch, P, d).
    """
    projection = torch.matmul(x, model.omega.T) + model.bias
    S = -model.scale * torch.sin(projection)
    grads = S.unsqueeze(-1) * model.omega.unsqueeze(0)
    return grads