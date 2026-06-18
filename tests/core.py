import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Theoretical Approximation: Wavelet / Smooth Layers
# (In practice, standard MLPs are used as proxies for the 
# strictly constrained Besov wavelet classes defined in the paper)
# ---------------------------------------------------------
class FunctionApproximator(nn.Module):
    """Proxy for the wavelet-parameterized functional classes (G, Phi, D)."""
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(), # Smooth activation to approximate higher-order regularity
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. Local Charts and Inverses (Section 3.1)
# ---------------------------------------------------------
class LocalChart(nn.Module):
    def __init__(self, d_dim, p_dim):
        super().__init__()
        # g_i: maps latent (d-dim) to ambient (p-dim)
        self.g = FunctionApproximator(in_dim=d_dim, out_dim=p_dim)
        # phi_i: maps ambient (p-dim) back to latent (d-dim)
        self.phi = FunctionApproximator(in_dim=p_dim, out_dim=d_dim)

# ---------------------------------------------------------
# 3. The Global Gluing Operator F^{g, \phi}
# ---------------------------------------------------------
class GeometricGluingOperator(nn.Module):
    def __init__(self, tau_radius=1.0):
        super().__init__()
        self.tau = tau_radius

    def smooth_cutoff_Gamma(self, radius):
        """
        Approximates the smooth cutoff function Gamma in H_C^{beta+1}.
        Returns 1 near the center, decaying smoothly to 0 at the boundary tau.
        """
        # Using a smooth sigmoid-based proxy for the theoretical bump function
        scaled = (radius - (self.tau * 0.8)) / (self.tau * 0.1)
        return 1.0 - torch.sigmoid(scaled)

    def forward(self, x_ambient, chart):
        """
        Applies F_i(x) = Gamma * g_i(phi_i(x)) + (1 - Gamma) * x
        """
        # 1. Project ambient point back to local latent space
        z_local = chart.phi(x_ambient)
        
        # 2. Calculate the distance from the chart origin (radius)
        radius = torch.norm(z_local, dim=-1, keepdim=True)
        
        # 3. Calculate smooth interpolation weight
        gamma_weight = self.smooth_cutoff_Gamma(radius)
        
        # 4. Project back to ambient space and interpolate
        x_projected = chart.g(z_local)
        x_glued = gamma_weight * x_projected + (1.0 - gamma_weight) * x_ambient
        
        return x_glued

# ---------------------------------------------------------
# 4. The Full Manifold Estimator
# ---------------------------------------------------------
class GenerativeManifoldEstimator(nn.Module):
    def __init__(self, num_charts, d_dim, p_dim, tau_radius=1.0):
        super().__init__()
        self.num_charts = num_charts
        self.d_dim = d_dim
        self.p_dim = p_dim
        
        # The collection of m local charts
        self.charts = nn.ModuleList([LocalChart(d_dim, p_dim) for _ in range(num_charts)])
        
        # The multinomial weights alpha (learnable parameters for chart selection)
        self.alpha_logits = nn.Parameter(torch.ones(num_charts))
        
        # The geometric gluing mechanism
        self.gluing_operator = GeometricGluingOperator(tau_radius)

    def get_alphas(self):
        """Returns the normalized probabilities of selecting each chart."""
        return F.softmax(self.alpha_logits, dim=0)

    def sample_latent(self, batch_size):
        """Samples from the truncated standard Gaussian gamma_d^n."""
        # For simplicity, using standard normal, but bounded in practice
        z = torch.randn(batch_size, self.d_dim)
        return z

    def generate(self, batch_size):
        """The complete Push-Forward Sampling Mechanism."""
        alphas = self.get_alphas()
        
        # 1. Sample which chart to use based on alpha
        chart_indices = torch.multinomial(alphas, batch_size, replacement=True)
        z = self.sample_latent(batch_size)
        
        x_generated = torch.zeros(batch_size, self.p_dim, device=z.device)
        
        # 2. Initial Push-Forward (g_i)
        for i in range(self.num_charts):
            mask = (chart_indices == i)
            if mask.sum() > 0:
                x_generated[mask] = self.charts[i].g(z[mask])
                
        # 3. Apply the global gluing operator F = F_m o ... o F_1
        for i in range(self.num_charts):
            x_generated = self.gluing_operator(x_generated, self.charts[i])
            
        return x_generated

# ---------------------------------------------------------
# 5. The Hölder Discriminator
# ---------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, p_dim):
        super().__init__()
        self.net = FunctionApproximator(in_dim=p_dim, out_dim=1)

    def forward(self, x):
        return self.net(x)