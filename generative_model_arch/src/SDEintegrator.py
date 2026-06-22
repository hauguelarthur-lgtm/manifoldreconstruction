import torch

def generate_samples(Z_0_list: list[torch.Tensor],
                     chart_models: list[torch.nn.Module],
                     precomputed_etas: list[list[torch.Tensor]], 
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> list[torch.Tensor]:
    """
    Executes Chart-Decoupled Intrinsic Flow Integration inside R^d.
    AMELIORATION V3:
    Accepts a chart-decoupled list of tailored Littlewood-Paley feature maps (`chart_models`),
    preventing size mismatch tensor collisions during drift evaluation.
    """
    num_charts = len(Z_0_list)
    U_t_list = [Z_0.clone().to(device) for Z_0 in Z_0_list]
    
    s = torch.linspace(0, 1.0, num_time_steps, device=device)
    time_grid = torch.sinh(s * 2.0) / torch.sinh(torch.tensor(2.0, device=device))
    
    dt_grid = torch.diff(time_grid)
    cfl_max_step_displacement = 0.50

    for step in range(num_time_steps - 1):
        t_val = time_grid[step]
        t_next = time_grid[step + 1]
        dt_nominal = dt_grid[step]
        
        t_score_denom = torch.clamp(t_val, min=1e-3)
        score_weight = (1.0 - t_val) / t_score_denom

        D_t_star = (1.0 - t_val)**2
        g_diff = torch.sqrt(torch.tensor(2.0 * D_t_star, device=device)) if (not ode_mode and D_t_star > 0) else torch.tensor(0.0, device=device)

        for i in range(num_charts):
            U_k = U_t_list[i]
            model_i = chart_models[i]
            
            if U_k.size(0) == 0 or model_i is None: continue

            eta_k = precomputed_etas[step][i].to(device)
            
            # --- STAGE 1: HEUN PREDICTOR ---
            b_k = torch.matmul(model_i(U_k), eta_k)
            
            if ode_mode:
                drift_k = b_k
            else:
                drift_k = b_k + score_weight * (t_val * b_k - U_k)

            max_drift_norm = torch.max(torch.norm(drift_k, dim=1))
            if max_drift_norm * dt_nominal > cfl_max_step_displacement:
                dt_effective = cfl_max_step_displacement / (max_drift_norm + 1e-8)
            else:
                dt_effective = dt_nominal

            noise_increment = torch.randn_like(U_k) * torch.sqrt(dt_effective)
            U_tilde = U_k + drift_k * dt_effective + (g_diff * noise_increment if not ode_mode else 0.0)

            # --- STAGE 2: HEUN CORRECTOR ---
            eta_next = precomputed_etas[step + 1][i].to(device)
            b_next = torch.matmul(model_i(U_tilde), eta_next)
            
            if ode_mode:
                drift_next = b_next
            else:
                t_next_denom = torch.clamp(t_next, min=1e-3)
                score_weight_next = (1.0 - t_next) / t_next_denom
                drift_next = b_next + score_weight_next * (t_next * b_next - U_tilde)

            drift_heun = 0.5 * (drift_k + drift_next)
            
            U_t_list[i] = U_k + drift_heun * dt_effective + (g_diff * noise_increment if not ode_mode else 0.0)

    return U_t_list