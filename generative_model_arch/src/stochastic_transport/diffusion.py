import torch

class KSISchedules:
    """
    Defines the exact interpolant kinematics and optimal diffusion schedules 
    for the continuous generative trajectory mapping z ~ N(0, I_p) to a ~ \mu^*(\tau).
    """
    def __init__(self, gamma_scale: float = 1.0, epsilon: float = 1e-7):
        """
        Args:
            gamma_scale: Scalar multiplier for the variance envelope \gamma_t.
            epsilon: Numerical clamp to prevent ZeroDivisionError at the boundary singularity t=0.
        """
        self.gamma_scale = gamma_scale
        self.epsilon = epsilon

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the noise-decay schedule.
        Strict boundary conditions: \alpha_0 = 1, \alpha_1 = 0.
        
        Args:
            t: Discretized time tensor in [0, 1].
        """
        return 1.0 - t

    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the data-activation schedule.
        Strict boundary conditions: \beta_0 = 0, \beta_1 = 1.
        
        Args:
            t: Discretized time tensor in [0, 1].
        """
        return t

    def dot_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the exact time derivative \dot{\alpha}_t.
        """
        return torch.full_like(t, -1.0)

    def dot_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the exact time derivative \dot{\beta}_t.
        """
        return torch.full_like(t, 1.0)

    def gamma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the variance envelope \gamma_t.
        Must vanish at boundaries (\gamma_0 = 0, \gamma_1 = 0) to ensure the SDE 
        endpoints collapse exactly to the target distributions.
        
        Args:
            t: Discretized time tensor in [0, 1].
        """
        return self.gamma_scale * t * (1.0 - t)

    def optimal_diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the theoretical optimal diffusion coefficient D_t^* minimizing the 
        path Kullback-Leibler divergence between the exact continuous interpolant 
        and the empirical finite-dimensional SDE.
        
        Formula: D_t^* = (\alpha_t * \gamma_t) / \beta_t
        
        Args:
            t: Discretized time tensor in [0, 1].
            
        Returns:
            D_t: The diffusion variance tensor.
        """
        a_t = self.alpha_t(t)
        b_t = self.beta_t(t)
        g_t = self.gamma_t(t)
        
        # Clamp beta_t to prevent division by zero at t=0
        b_t_clamped = torch.clamp(b_t, min=self.epsilon)
        
        D_t_star = (a_t * g_t) / b_t_clamped
        return D_t_star

    def evaluate_all_schedules(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Evaluates and returns all kinematic parameters for a given time step t
        to optimize batch processing during the empirical linear system assembly.
        """
        return {
            'alpha': self.alpha_t(t),
            'beta': self.beta_t(t),
            'dot_alpha': self.dot_alpha_t(t),
            'dot_beta': self.dot_beta_t(t),
            'gamma': self.gamma_t(t),
            'D_star': self.optimal_diffusion_coefficient(t)
        }