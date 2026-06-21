import torch

def generate_samples(Z_0_list: list,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> list:
    """
    Executes Chart-Decoupled Intrinsic Flow Integration inside R^d (arXiv:2602.20070).
    Integrates independent chart batches without invalid tangent bundle mixing.
    Stripped of console-spamming loop prints and permissive overshoot thresholds.
    """
    num_charts = len(Z_0_list)
    U_t_list = [Z_0.clone().to(device) for Z_0 in Z_0_list]
    model.eval()
    
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    max_drift = 2.0  # Clamped from 15.0 to prevent ballistic trajectory overshoot

    time_grid = torch.linspace(0, 1.0 - epsilon_num, num_time_steps, device=device)
    dt_tensor = torch.tensor(dt, device=device)

    for step in range(num_time_steps):
        t_val = time_grid[step]
        D_t_star = (1.0 - t_val) ** 2
        sqrt_2D = torch.sqrt(torch.tensor(2.0 * D_t_star, device=device)) if D_t_star > 0 else torch.tensor(0.0, device=device)

        for i in range(num_charts):
            U_t = U_t_list[i]
            if U_t.size(0) == 0:
                continue
                
            eta_t_i = precomputed_etas[step][i].to(device)
            features = model(U_t)                  # (Batch, P)
            b_t = torch.matmul(features, eta_t_i)  # (Batch, d)
            
            if ode_mode:
                drift_norms = torch.norm(b_t, dim=1, keepdim=True)
                b_t_clamped = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                U_t_list[i] = U_t + b_t_clamped * dt 
            else:
                # Closed-form algebraic score mapping (arXiv:2602.20070)
                s_t = (t_val * b_t - U_t) / torch.clamp(1.0 - t_val, min=1e-8)
                full_drift = b_t + D_t_star * s_t
                
                drift_norms = torch.norm(full_drift, dim=1, keepdim=True)
                full_drift = full_drift * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                dW = torch.randn_like(U_t) * torch.sqrt(dt_tensor)
                U_t_list[i] = U_t + full_drift * dt + sqrt_2D * dW

    # Terminal Epsilon Boundary Step
    for i in range(num_charts):
        U_t = U_t_list[i]
        if U_t.size(0) == 0:
            continue
        eta_final_i = precomputed_etas[-1][i].to(device)
        b_t_final = torch.matmul(model(U_t), eta_final_i)
        U_t_list[i] = (U_t + b_t_final * epsilon_num).detach()
        
    return U_t_list