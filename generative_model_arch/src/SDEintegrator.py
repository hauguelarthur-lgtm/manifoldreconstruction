import math
import torch
from src.wavelet_map import compute_feature_gradients

def generate_samples(Z_0_list: list,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> list:
    """
    Executes Chart-Decoupled Intrinsic Flow Integration inside R^d.
    Integrates latent particles from N(0, I_d) along intrinsic Whitney streamlines.
    """
    num_charts = len(Z_0_list)
    U_t_list = [Z_0.clone().to(device) for Z_0 in Z_0_list]
    model.eval()
    
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    max_drift = 15.0

    # Synchronization-free temporal grid
    time_grid = [step * dt for step in range(num_time_steps)]

    for step, t_val in enumerate(time_grid):
        D_t_star = (1.0 - t_val) ** 2
        sqrt_2D = math.sqrt(2.0 * D_t_star) if D_t_star > 0 else 0.0

        for i in range(num_charts):
            U_t = U_t_list[i]
            if U_t.size(0) == 0:
                continue
                
            eta_t_i = precomputed_etas[step][i].to(device)
            
            # Predictor Drift Computation strictly inside R^d
            feature_grads = compute_feature_gradients(model, U_t)
            b_t = torch.matmul(feature_grads.transpose(1, 2), eta_t_i)

            if ode_mode:
                # Heun's Method (2nd Order Deterministic Probability Flow)
                drift_norms = torch.norm(b_t, dim=1, keepdim=True)
                b_t_clamped = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                U_pred = U_t + b_t_clamped * dt 
                if step < num_time_steps - 1:
                    eta_t_next_i = precomputed_etas[step + 1][i].to(device)
                    feature_grads_next = compute_feature_gradients(model, U_pred)
                    b_t_next = torch.matmul(feature_grads_next.transpose(1, 2), eta_t_next_i)
                    
                    drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                    b_t_next_clamped = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                    
                    U_t_list[i] = U_t + 0.5 * (b_t_clamped + b_t_next_clamped) * dt
                else:
                    U_t_list[i] = U_pred
            else:
                # Euler-Maruyama Method (Stochastic KSI Framework inside R^d)
                s_t = (t_val * b_t - U_t) / max(1.0 - t_val, 1e-8)
                full_drift = b_t + D_t_star * s_t
                
                # Unbounded drift clamp applied to full resultant drift vector
                drift_norms = torch.norm(full_drift, dim=1, keepdim=True)
                full_drift = full_drift * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                dW = torch.randn_like(U_t) * math.sqrt(dt)
                U_t_list[i] = U_t + full_drift * dt + sqrt_2D * dW

    # Terminal Boundary Projection (Epsilon gap closure strictly in R^d)
    for i in range(num_charts):
        U_t = U_t_list[i]
        if U_t.size(0) == 0:
            continue
            
        eta_final_i = precomputed_etas[-1][i].to(device)
        feature_grads_final = compute_feature_gradients(model, U_t)
        b_t_final = torch.matmul(feature_grads_final.transpose(1, 2), eta_final_i)
        
        U_t_list[i] = (U_t + b_t_final * epsilon_num).detach()
        
    return U_t_list