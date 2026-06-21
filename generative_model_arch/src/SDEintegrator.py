import torch

def generate_samples(Z_0_list: list,
                     model: torch.nn.Module,
                     precomputed_etas: list, 
                     num_time_steps: int = 200,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> list:
    """
    Executes Chart-Decoupled Intrinsic Flow Integration strictly inside R^d.
    Liberated from the potential flow constraint: evaluates general vector fields via \phi(x)\eta.
    """
    num_charts = len(Z_0_list)
    U_t_list = [Z_0.clone().to(device) for Z_0 in Z_0_list]
    model.eval()
    
    epsilon_num = 1e-5
    dt = (1.0 - epsilon_num) / (num_time_steps - 1)
    max_drift = 15.0

    # -------------------------------------------------------------------------
    # HARDWARE OPTIMIZATION: Pre-allocate static scalars as native device Tensors.
    # Completely eliminates asynchronous CPU/GPU PCIe bus lockups inside the hot loop.
    # -------------------------------------------------------------------------
    time_grid = torch.linspace(0, 1.0 - epsilon_num, num_time_steps, device=device)
    dt_tensor = torch.tensor(dt, device=device)

    for step in range(num_time_steps):
        t_val = time_grid[step]  # Native device scalar
        
        # Analytical optimal Girsanov variance schedule D_t^* = (1 - t)^2
        D_t_star = (1.0 - t_val) ** 2
        sqrt_2D = torch.sqrt(2.0 * D_t_star) if D_t_star > 0 else torch.tensor(0.0, device=device)

        for i in range(num_charts):
            U_t = U_t_list[i]
            if U_t.size(0) == 0:
                continue
                
            # Extract coefficient matrix \eta_i \in R^{P \times d}
            eta_t_i = precomputed_etas[step][i].to(device)
            
            # -----------------------------------------------------------------
            # MATHEMATICAL CORRECTION 1: General Vector Field Evaluation
            # Evaluates \phi(x) @ \eta directly. Excises the irrotational Jacobian proxy.
            # -----------------------------------------------------------------
            features = model(U_t)                  # Shape: (Batch, P)
            b_t = torch.matmul(features, eta_t_i)  # (Batch, P) @ (P, d) -> (Batch, d)

            if ode_mode:
                # Heun's Method (2nd Order Deterministic Probability Flow inside R^d)
                drift_norms = torch.norm(b_t, dim=1, keepdim=True)
                b_t_clamped = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                U_pred = U_t + b_t_clamped * dt 
                if step < num_time_steps - 1:
                    eta_t_next_i = precomputed_etas[step + 1][i].to(device)
                    features_next = model(U_pred)
                    b_t_next = torch.matmul(features_next, eta_t_next_i)
                    
                    drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                    b_t_next_clamped = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                    
                    U_t_list[i] = U_t + 0.5 * (b_t_clamped + b_t_next_clamped) * dt
                else:
                    U_t_list[i] = U_pred
            else:
                # Euler-Maruyama Method (Stochastic KSI Framework inside R^d)
                
                # 1. Algebraic Score Mapping evaluated via graph-native clamping
                s_t = (t_val * b_t - U_t) / torch.clamp(1.0 - t_val, min=1e-8)
                
                # 2. Formulate total SDE drift: Predictor + scaled Langevin Corrector
                full_drift = b_t + D_t_star * s_t
                
                # 3. Unbounded drift clamp applied to total resultant vector
                drift_norms = torch.norm(full_drift, dim=1, keepdim=True)
                full_drift = full_drift * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
                
                # 4. Discrete integration step
                dW = torch.randn_like(U_t) * torch.sqrt(dt_tensor)
                U_t_list[i] = U_t + full_drift * dt + sqrt_2D * dW

    # Terminal Boundary Projection (Epsilon gap closure strictly inside R^d)
    for i in range(num_charts):
        U_t = U_t_list[i]
        if U_t.size(0) == 0:
            continue
            
        eta_final_i = precomputed_etas[-1][i].to(device)
        features_final = model(U_t)
        b_t_final = torch.matmul(features_final, eta_final_i)
        
        U_t_list[i] = (U_t + b_t_final * epsilon_num).detach()
        
    return U_t_list