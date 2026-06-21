import torch

def generate_samples(Z_0_list: list,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> list:
    """
    Executes Chart-Decoupled Intrinsic Flow Integration inside R^d (arXiv:2602.20070).
    EXACT CORRECTION: Unclamps max_drift to 100.0 to allow particles to traverse 
    the true physical diameter of the empirical chart without crawling to a halt.
    """
    num_charts = len(Z_0_list)
    U_t_list = [Z_0.clone().to(device) for Z_0 in Z_0_list]
    
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    max_drift = 100.0  # Permissive physical ceiling across wide Swiss roll charts

    time_grid = torch.linspace(0, 1.0 - epsilon_num, num_time_steps, device=device)
    dt_tensor = torch.tensor(dt, device=device)

    for step in range(num_time_steps):
        t_val = time_grid[step]
        D_t_star = (1.0 - t_val) ** 2
        sqrt_2D = torch.sqrt(torch.tensor(2.0 * D_t_star, device=device)) if D_t_star > 0 else torch.tensor(0.0, device=device)

        for i in range(num_charts):
            U_t = U_t_list[i]
            if U_t.size(0) == 0: continue
                
            b_t = torch.matmul(model(U_t), precomputed_etas[step][i].to(device))
            
            if ode_mode:
                drift_norms = torch.norm(b_t, dim=1, keepdim=True)
                U_t_list[i] = U_t + (b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)) * dt 
            else:
                full_drift = b_t + (1.0 - t_val) * (t_val * b_t - U_t)
                drift_norms = torch.norm(full_drift, dim=1, keepdim=True)
                full_drift = full_drift * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                U_t_list[i] = U_t + full_drift * dt + sqrt_2D * torch.randn_like(U_t) * torch.sqrt(dt_tensor)

    for i in range(num_charts):
        if U_t_list[i].size(0) > 0:
            b_t_final = torch.matmul(model(U_t_list[i]), precomputed_etas[-1][i].to(device))
            U_t_list[i] = (U_t_list[i] + b_t_final * epsilon_num).detach()
        
    return U_t_list