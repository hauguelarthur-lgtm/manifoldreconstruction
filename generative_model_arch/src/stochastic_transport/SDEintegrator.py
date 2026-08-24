import torch
import math
from typing import Callable, Union

class SDEIntegrator:
    """
    Executes the continuous generative generation by integrating the optimal 
    Stochastic Differential Equation (SDE) utilizing the deterministic drift 
    coefficients and the algebraically constrained topological feature maps.
    """
    def __init__(self, 
                 schedules, 
                 num_steps: int = 100, 
                 t_eps: float = 1e-4):
        """
        Args:
            schedules: Instance of KSISchedules to evaluate continuous kinematics.
            num_steps: Number of temporal discretization steps for the SDE integration.
            t_eps: Minimum integration time boundary. Explicitly bypasses the singularity 
                   at t=0 where the analytical drift term (\dot{\beta}_t / \beta_t) diverges 
                   to infinity.
        """
        self.schedules = schedules
        self.num_steps = num_steps
        self.t_eps = t_eps

    def integrate(self, 
                  eta_t_sequence: torch.Tensor, 
                  feature_extractor, 
                  atlas: list[dict], 
                  partition_of_unity_evaluator: Callable,
                  B: int, 
                  p: int, 
                  device: torch.device) -> torch.Tensor:
        """
        Integrates the continuous generative SDE from t = t_eps to t = 1.0 to map 
        base Gaussian noise mathematically onto the target manifold density \mu^*(\tau).
        
        Args:
            eta_t_sequence: The solved optimal drift coefficients for each time step.
                            Shape: (num_steps, P) where P = m * d.
            feature_extractor: Instance of FeatureMapExtractor.
            atlas: The frozen multi-order topological atlas.
            partition_of_unity_evaluator: Function yielding normalized C^\infty bump weights.
            B: Number of macroscopic samples to generate (Batch size).
            p: The ambient data dimension.
            device: Target computational device.
            
        Returns:
            X_t: The physically valid generated states on the manifold \mathcal{M}^*.
                 Shape: (B, p).
        """
        # 1. Base Prior Initialization
        # Z ~ \mathcal{N}(0, I_p)
        X_t = torch.randn((B, p), device=device)
        
        # Temporal step size restricted to the defined safe integration window
        dt = (1.0 - self.t_eps) / self.num_steps
        
        for step in range(self.num_steps):
            # Evaluate exact macroscopic time to maintain schedule alignment
            t_val = self.t_eps + step * dt
            t_tensor = torch.tensor([t_val], device=device)
            
            # 2. Extract Continuous Interpolant Kinematics
            scheds = self.schedules.evaluate_all_schedules(t_tensor)
            beta_t = scheds['beta'].item()
            dot_beta_t = scheds['dot_beta'].item()
            D_star_t = scheds['D_star'].item()
            
            eta_current = eta_t_sequence[step]
            
            # 3. Dynamic Feature Map Reconstruction
            # Extracts the exact localized orthogonal projectors \nabla\phi(X_t)
            # evaluated at the current spatial coordinates of the trajectory.
            # Shape: (B, p, P)
            Phi_t = feature_extractor.construct_global_basis(X_t, 
                                                             atlas, 
                                                             partition_of_unity_evaluator)
            
            # 4. Drift Vector Assembly
            # \hat{b}_t(X_t) = \nabla\phi(X_t) \eta_t(\tau)
            # Phi_t is (B, p, P), eta_current is (P,) -> output is (B, p)
            b_hat_t = torch.matmul(Phi_t, eta_current.unsqueeze(1)).squeeze(2)
            
            # Formulate the mathematically exact continuous transport structural drift.
            # The strictly algebraic (dot_beta_t / beta_t) penalty keeps the trajectories
            # geometrically tethered as they approach the invariant support.
            structural_drift = 2.0 * b_hat_t - (dot_beta_t / beta_t) * X_t
            
            # 5. Optimal Stochastic Mitigation
            # Generates the Brownian motion increments dW_t
            dW_t = torch.randn_like(X_t) * math.sqrt(dt)
            
            # Applies the Girsanov-derived variance bound to statistically absorb 
            # truncation errors triggered by the finite k-degree Taylor expansion.
            diffusion_term = math.sqrt(2.0 * D_star_t) * dW_t
            
            # 6. SDE Integration Step (Euler-Maruyama)
            # For heavily non-linear topologies, this linear integration step relies strictly
            # on the optimal diffusion term to prevent trajectories from exiting the support.
            X_t = X_t + structural_drift * dt + diffusion_term
            
        return X_t