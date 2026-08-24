import torch

class KSILinearSolver:
    """
    Executes the deterministic macroscopic linear regression to compute the time-dependent
    drift coefficients \eta_t(\tau) for the Kernelized Stochastic Interpolant.
    """
    def __init__(self, lambda_ksi: float = 1e-5):
        """
        Args:
            lambda_ksi: Macroscopic Tikhonov regularization penalty to guarantee 
                        strict positive-definiteness under temporal density shifts.
        """
        self.lambda_ksi = lambda_ksi

    def compute_interpolant_kinematics(self, 
                                       A: torch.Tensor, 
                                       Z: torch.Tensor, 
                                       alpha_t: float, 
                                       beta_t: float, 
                                       dot_alpha_t: float, 
                                       dot_beta_t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the empirical stochastic interpolant state and its exact time derivative.
        
        Args:
            A: Empirical data batch drawn from \mu^*(\tau) of shape (N, p)
            Z: Isotropic Gaussian noise batch \mathcal{N}(0, I_p) of shape (N, p)
            alpha_t, beta_t: Interpolant schedule scalars at time t
            dot_alpha_t, dot_beta_t: Schedule time derivatives at time t
            
        Returns:
            I_t: Interpolant state tensor of shape (N, p)
            dot_I_t: Interpolant velocity tensor of shape (N, p)
        """
        I_t = alpha_t * Z + beta_t * A
        dot_I_t = dot_alpha_t * Z + dot_beta_t * A
        
        return I_t, dot_I_t

    def solve_eta(self, 
                  Phi_t: torch.Tensor, 
                  dot_I_t: torch.Tensor) -> torch.Tensor:
        """
        Assembles the P x P macroscopic Gram matrix and target vector via batched tensor 
        operations, applies topological regularization, and solves for the deterministic 
        drift coefficients \eta_t.
        
        Args:
            Phi_t: Globally glued feature map tensor \nabla\phi(I_t) of shape (N, p, P)
            dot_I_t: Interpolant velocity tensor of shape (N, p)
            
        Returns:
            eta_t: Optimal generative drift parameter vector of shape (P,)
        """
        N, p, P = Phi_t.shape
        device = Phi_t.device
        
        # 1. Batched Tensorial Gram Matrix Assembly
        # Phi_t^T is (N, P, p)
        Phi_t_T = Phi_t.transpose(1, 2)
        
        # \Phi_t^T \Phi_t yields (N, P, P) -> mean over N yields \hat{K}_t of shape (P, P)
        K_t_batch = torch.bmm(Phi_t_T, Phi_t)
        K_t = K_t_batch.mean(dim=0)
        
        # 2. Empirical Target Vector Assembly
        # dot_I_t is unsqueezed to (N, p, 1) for batched matrix multiplication
        # \Phi_t^T \dot{I}_t yields (N, P, 1) -> mean over N yields (P, 1) -> squeeze to (P,)
        r_t_batch = torch.bmm(Phi_t_T, dot_I_t.unsqueeze(2))
        r_t = r_t_batch.mean(dim=0).squeeze(1)
        
        # 3. Macroscopic Tikhonov Regularization
        # \hat{K}_t^{reg} = \hat{K}_t + \lambda_{KSI} I_P
        K_t_reg = K_t + torch.eye(P, device=device) * self.lambda_ksi
        
        # 4. Deterministic System Resolution
        # Solve \hat{K}_t^{reg} \eta_t = \hat{r}_t
        try:
            eta_t = torch.linalg.solve(K_t_reg, r_t)
        except (RuntimeError, torch._C._LinAlgError):
            # Fallback for degenerate conditioning limits
            K_t_pinv = torch.linalg.pinv(K_t_reg, hermitian=True)
            eta_t = torch.matmul(K_t_pinv, r_t)
            
        return eta_t