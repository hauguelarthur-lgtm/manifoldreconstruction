import torch
import math

def construct_whitney_atlas(data: torch.Tensor, 
                            intrinsic_dim: int,
                            num_charts: int | str = None,
                            target_beta: float = 1.50,
                            packing_multiplier: float = 3.0) -> tuple:
    """
    Constructs an Overlapping Submanifold Atlas (arXiv:2506.19587).
    IMMUTABLE CORRECTION: Locks target_beta permanently to the Benamou-Brenier 
    physical invariant \beta = 1.50, entirely excising the estimation trapdoor.
    """
    N, p = data.shape
    d = int(intrinsic_dim)
    device = data.device
    
    # HARD PHYSICAL CONSTANT OF THE STOCHASTIC INTERPOLANT
    resolved_beta = 1.50 

    if num_charts is None or num_charts == 'auto' or num_charts == 'none' or num_charts == 0:
        minimax_exponent = float(d) / (2.0 * resolved_beta + float(d))
        m = max(int(math.ceil(packing_multiplier * math.pow(N, minimax_exponent))), d + 2)
    else:
        m = int(num_charts)

    fps_centers = torch.zeros(m, p, device=device)
    fps_centers[0] = data[torch.randint(0, N, (1,))]
    distances = torch.cdist(data, fps_centers[0].unsqueeze(0)).squeeze(1)

    for i in range(1, m):
        fps_centers[i] = data[torch.argmax(distances)]
        distances = torch.minimum(distances, torch.cdist(data, fps_centers[i].unsqueeze(0)).squeeze(1))

    delta = torch.max(distances).item()
    chart_radius = 1.5 * delta  
    membership_mask = torch.cdist(data, fps_centers) < chart_radius

    atlas_frames, intrinsic_coords, chart_ambient_indices = [], [], []

    for i in range(m):
        chart_idx = torch.nonzero(membership_mask[:, i]).squeeze(1)
        chart_ambient_indices.append(chart_idx.cpu())

        X_i = data[chart_idx]
        N_i = X_i.size(0)
        quad_dim = d * (d + 1) // 2
        if N_i < quad_dim: raise ValueError(f"Chart {i} too sparse ({N_i} pts) for Weingarten regression.")

        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i

        cov_i = torch.matmul(centered_X.T, centered_X) / (N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        Q_i = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:d]]

        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        N_err = centered_X - torch.matmul(U_i, Q_i.T)  
        U_quad = torch.zeros(N_i, quad_dim, device=device)
        col = 0
        for dim1 in range(d):
            for dim2 in range(dim1, d):
                U_quad[:, col] = U_i[:, dim1] * U_i[:, dim2]
                col += 1

        U_quad_mean = U_quad.mean(dim=0, keepdim=True)
        U_quad_std = U_quad.std(dim=0, keepdim=True) + 1e-8
        U_quad_norm = (U_quad - U_quad_mean) / U_quad_std

        # Rigorous normal bundle scaling strictly by normal rank (p - d)
        global_n_err_std = torch.sqrt(torch.sum(torch.var(N_err, dim=0)) / float(p - d)) + 1e-6
        N_err_norm = N_err / global_n_err_std

        G = torch.matmul(U_quad_norm.T, U_quad_norm)
        lambda_reg = 1e-4 * (torch.trace(G) / quad_dim) + 1e-7
        
        W_norm = torch.linalg.solve(G + torch.eye(quad_dim, device=device) * lambda_reg, torch.matmul(U_quad_norm.T, N_err_norm))
        W_i = (W_norm / U_quad_std.T) * global_n_err_std
        
        atlas_frames.append({'mu': (mu_i - torch.matmul(U_quad_mean.squeeze(0), W_i)).cpu(), 'Q': Q_i.cpu(), 'W': W_i.cpu()})

    return membership_mask.cpu(), intrinsic_coords, atlas_frames, chart_ambient_indices