import torch
from src.gluing import compute_subordinated_partition_of_unity

def generate_samples(Z_0: torch.Tensor,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     centroids: torch.Tensor,
                     chart_radii: torch.Tensor,
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> torch.Tensor:
    """
    Executes Overlapping Intrinsic Flow Integration inside R^d.
    Continuously blends local Besov vector fields via compact bump partition of unity.
    """
    U_t = Z_0.clone().to(device)
    centroids = centroids.to(device)
    chart_radii = chart_radii.to(device)
    model.eval()
    
    m = centroids.shape[0]
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    max_drift = 15.0

    time_grid = torch.linspace(0, 1.0 - epsilon_num, num_time_steps, device=device)
    dt_tensor = torch.tensor(dt, device=device)

    for step in range(num_time_steps):
        t_val = time_grid[step]
        D_t_star = (1.0 - t_val) ** 2
        sqrt_2D = torch.sqrt(torch.tensor(2.0 * D_t_star, device=device)) if D_t_star > 0 else 0.0

        # Evaluate compact partition of unity weights \xi_i(U_t) dynamically in R^d
        weights = compute_subordinated_partition_of_unity(U_t, centroids, chart_radii)
        
        # Evaluate global blended vector field across all active charts
        features = model(U_t)  # (Batch, P)
        b_t_blended = torch.zeros_like(U_t)

        for i in range(m):
            eta_t_i = precomputed_etas[step][i].to(device)
            b_t_local = torch.matmul(features, eta_t_i)  # (Batch, d)
            rho_i = weights[:, i].unsqueeze(1)
            b_t_blended += rho_i * b_t_local

        if ode_mode:
            drift_norms = torch.norm(b_t_blended, dim=1, keepdim=True)
            b_t_clamped = b_t_blended * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
            U_t = U_t + b_t_clamped * dt 
        else:
            # Score-Corrected SDE step natively in R^d
            s_t = (t_val * b_t_blended - U_t) / torch.clamp(1.0 - t_val, min=1e-8)
            full_drift = b_t_blended + D_t_star * s_t
            
            drift_norms = torch.norm(full_drift, dim=1, keepdim=True)
            full_drift = full_drift * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
            
            dW = torch.randn_like(U_t) * torch.sqrt(dt_tensor)
            U_t = U_t + full_drift * dt + sqrt_2D * dW

    # Terminal Epsilon Step
    features_final = model(U_t)
    b_final_blended = torch.zeros_like(U_t)
    weights_final = compute_subordinated_partition_of_unity(U_t, centroids, chart_radii)
    
    for i in range(m):
        eta_final_i = precomputed_etas[-1][i].to(device)
        rho_i = weights_final[:, i].unsqueeze(1)
        b_final_blended += rho_i * torch.matmul(features_final, eta_final_i)
        
    U_t = U_t + b_final_blended * epsilon_num
    return U_t.detach()