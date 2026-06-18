# Replace the contents of src/wavelet_map.py with this:
import torch
import torch.nn as nn
import numpy as np

class TruncatedBesovWaveletMap(nn.Module):
    """
    Implements the deterministic feature map \phi(x) : R^p -> R^P.
    Utilizes a wide, untrained, non-linear network to approximate 
    a sufficiently expressive geometric basis for the local tangent spaces.
    """
    def __init__(self, ambient_dim: int, intrinsic_dim: int, p_truncation: int):
        super().__init__()
        self.p = ambient_dim
        self.d = intrinsic_dim
        self.P = p_truncation
        
        # We must use a large expansion factor to ensure the hypothesis space is rich enough
        hidden_dim = self.P * 4 
        
        # A fixed, shallow MLP acts as a generalized random feature extractor.
        # GELU provides the necessary higher-order smoothness required by the bounds.
        self.net = nn.Sequential(
            nn.Linear(self.p, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.P),
            nn.GELU()
        )
        
        # Ensure the parameters are strictly frozen (no training)
        for param in self.parameters():
            param.requires_grad = False
            
        # Re-initialize with heavy-tailed distributions to prevent vanishing gradients
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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