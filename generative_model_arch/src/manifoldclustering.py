import torch

def construct_whitney_atlas(data: torch.Tensor, 
                            num_charts: int, 
                            intrinsic_dim: int) -> tuple:
    N, p = data.shape
    d = intrinsic_dim
    device = data.device

    fps_centers = torch.zeros(num_charts, p, device=device)
    fps_centers[0] = data[torch.randint(0, N, (1,))]
    distances = torch.cdist(data, fps_centers[0].unsqueeze(0)).squeeze(1)

    for i in range(1, num_charts):
        farthest_idx = torch.argmax(distances)
        fps_centers[i] = data[farthest_idx]
        dist_to_new = torch.cdist(data, fps_centers[i].unsqueeze(0)).squeeze(1)
        distances = torch.minimum(distances, dist_to_new)

    chart_radius = 1.5 * torch.max(distances).item()
    membership_mask = torch.cdist(data, fps_centers) < chart_radius

    atlas_frames = []
    intrinsic_coords = []
    chart_ambient_indices = []

    for i in range(num_charts):
        in_chart = membership_mask[:, i]
        chart_idx = torch.nonzero(in_chart).squeeze(1)
        chart_ambient_indices.append(chart_idx.cpu())

        X_i = data[chart_idx]
        N_i = X_i.size(0)

        quad_dim = d * (d + 1) // 2
        if N_i < quad_dim:
            raise ValueError(f"Chart {i} population ({N_i}) is too sparse to solve Weingarten tensor.")

        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i

        cov_i = torch.matmul(centered_X.T, centered_X) / (N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        Q_i = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:d]]

        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        # STEP 2: Weingarten Regression via Global Isotropic Standardization
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

        # MATHEMATICAL FIX 1: Global Isotropic Residual Standardization
        # Eliminates zero-division noise amplification on degenerate normal axes.
        global_n_err_std = torch.sqrt(torch.var(N_err, dim=0).mean()) + 1e-6
        N_err_norm = N_err / global_n_err_std

        G = torch.matmul(U_quad_norm.T, U_quad_norm)
        lambda_reg = 1e-4 * (torch.trace(G) / quad_dim) + 1e-7
        G_reg = G + torch.eye(quad_dim, device=device) * lambda_reg
        
        W_norm = torch.linalg.solve(G_reg, torch.matmul(U_quad_norm.T, N_err_norm))
        W_i = (W_norm / U_quad_std.T) * global_n_err_std

        # MATHEMATICAL FIX 2: Intercept-Shifted Base Centroid \mu_i^*
        mu_i_star = mu_i - torch.matmul(U_quad_mean.squeeze(0), W_i)

        print(f"Chart {i:02d} Weingarten Curvature Norm ||W_i||: {W_i.norm().item():.4f}")

        atlas_frames.append({
            'mu': mu_i_star.cpu(), 
            'Q': Q_i.cpu(), 
            'W': W_i.cpu()
        })

    return membership_mask.cpu(), intrinsic_coords, atlas_frames, chart_ambient_indices