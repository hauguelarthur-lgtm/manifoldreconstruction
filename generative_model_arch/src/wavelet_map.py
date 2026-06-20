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
        
        # Initialize placeholder omegas (will be strictly overridden by calibrate())
        self.omega = nn.Parameter(torch.randn(self.P, self.p), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(self.P) * 2 * np.pi, requires_grad=False)
        self.scale = float(np.sqrt(2.0 / self.P))

    def calibrate(self, x: torch.Tensor, subsample: int = 2000):
        """
        Dynamically calibrates the Random Fourier Feature bandwidths using the 
        median pairwise distance heuristic of the empirical data manifold.
        """
        with torch.no_grad():
            # 1. Subsample to mathematically bound O(N^2) memory complexity during distance computation
            N = x.size(0)
            if N > subsample:
                indices = torch.randperm(N, device=x.device)[:subsample]
                x_sub = x[indices]
            else:
                x_sub = x
            
            # 2. Compute exact L2 pairwise distances
            distances = torch.pdist(x_sub, p=2)
            
            # 3. Extract the median spatial scale
            median_dist = torch.median(distances).item()
            if median_dist <= 1e-8:
                median_dist = 1.0  # Failsafe limit for structurally collapsed geometries
            
            # 4. In the Fourier domain, frequency is inversely proportional to spatial distance
            base_freq = 1.0 / median_dist
            
            # 5. Distribute frequencies across three dynamic spectral bands (Coarse, Medium, Fine)
            scales = [base_freq * 0.5, base_freq, base_freq * 2.0]
            p_per_scale = self.P // len(scales)
            
            omegas = []
            for scale in scales:
                # Isotropic Gaussian sampling scaled by the optimal frequency and ambient dimension
                w = torch.randn(p_per_scale, self.p, device=x.device) * (scale / np.sqrt(self.p))
                omegas.append(w)
                
            # Handle non-divisible polynomial remainders
            if self.P % len(scales) != 0:
                rem = self.P % len(scales)
                omegas.append(torch.randn(rem, self.p, device=x.device) * (base_freq / np.sqrt(self.p)))
                
            # Update the parameter tensor strictly in-place to preserve device mapping
            self.omega.copy_(torch.cat(omegas, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.omega.T) + self.bias
        return self.scale * torch.cos(projection)

def compute_feature_gradients(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    projection = torch.matmul(x, model.omega.T) + model.bias
    S = -model.scale * torch.sin(projection)
    grads = S.unsqueeze(-1) * model.omega.unsqueeze(0)
    return grads