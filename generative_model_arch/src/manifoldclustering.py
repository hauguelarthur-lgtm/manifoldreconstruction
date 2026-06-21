import torch

def construct_whitney_atlas(data: torch.Tensor, 
                            num_charts: int, 
                            intrinsic_dim: int) -> tuple:
    """
    Constructs a true Stéphanovitch / Fefferman Overlapping Submanifold Atlas (arXiv:2506.19587).
    Resolves the 1st-order geometric flattening trap by enforcing barycentric chart centering
    and dual scale-invariant standardization during Weingarten tensor regression.
    """
    N, p = data.shape
    d = intrinsic_dim
    device = data.device

    # ---------------------------------------------------------
    # STEP 1: Greedy Farthest Point Sampling (FPS) Delta-Net
    # Establishes the maximal separation net centers c_i
    # ---------------------------------------------------------
    fps_centers = torch.zeros(num_charts, p, device=device)
    fps_centers[0] = data[torch.randint(0, N, (1,))]
    distances = torch.cdist(data, fps_centers[0].unsqueeze(0)).squeeze(1)

    for i in range(1, num_charts):
        farthest_idx = torch.argmax(distances)
        fps_centers[i] = data[farthest_idx]
        dist_to_new = torch.cdist(data, fps_centers[i].unsqueeze(0)).squeeze(1)
        distances = torch.minimum(distances, dist_to_new)

    # Covering radius delta is the maximum distance any data point has to its nearest FPS center
    delta = torch.max(distances).item()
    chart_radius = 1.5 * delta  # 1.5 multiplier enforces smooth open overlapping covering

    all_pairwise_dists = torch.cdist(data, fps_centers)
    membership_mask = all_pairwise_dists < chart_radius

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
            raise ValueError(f"Chart {i} population ({N_i}) is too sparse to solve {quad_dim}-dimensional Weingarten tensor.")

        # ---------------------------------------------------------
        # MATHEMATICAL CORRECTION 1: True Barycentric Centering
        # Anchor the chart strictly at the empirical mean of the patch,
        # guaranteeing zero-mean 1st-order intrinsic coordinates U_i.
        # ---------------------------------------------------------
        mu_i = X_i.mean(dim=0)
        centered_X = X_i - mu_i

        # 1st-Order Tangent Plane via PCA
        cov_i = torch.matmul(centered_X.T, centered_X) / (N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        top_indices = torch.argsort(eigenvalues, descending=True)[:d]
        Q_i = eigenvectors[:, top_indices]  # (p x d)

        # Intrinsic Coordinates U_i \in R^{N_i \times d} (strictly zero-mean)
        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        # ---------------------------------------------------------
        # STEP 2: 2nd-Order Weingarten Curvature Regression
        # Overcomes the flat beta=2 ceiling by fitting local normal paraboloids
        # ---------------------------------------------------------
        # Normal-bundle residuals: N_err = X_centered - U_i @ Q_i.T
        N_err = centered_X - torch.matmul(U_i, Q_i.T)  # (N_i x p)

        # Formulate upper-triangular quadratic outer products
        U_quad = torch.zeros(N_i, quad_dim, device=device)
        col = 0
        for dim1 in range(d):
            for dim2 in range(dim1, d):
                U_quad[:, col] = U_i[:, dim1] * U_i[:, dim2]
                col += 1

        # ---------------------------------------------------------
        # MATHEMATICAL CORRECTION 2: Dual Unit-Variance Standardization
        # Standardize both design matrix and target to unit variance
        # before solving the regularized system, preventing underflow.
        # ---------------------------------------------------------
        U_quad_mean = U_quad.mean(dim=0, keepdim=True)
        U_quad_std = U_quad.std(dim=0, keepdim=True) + 1e-8
        U_quad_norm = (U_quad - U_quad_mean) / U_quad_std

        N_err_mean = N_err.mean(dim=0, keepdim=True)
        N_err_std = N_err.std(dim=0, keepdim=True) + 1e-8
        N_err_norm = (N_err - N_err_mean) / N_err_std

        # Form Gram matrix on normalized quadratic features
        G = torch.matmul(U_quad_norm.T, U_quad_norm)
        
        # Adaptive trace-scaled Tikhonov penalty
        alpha = 1e-4
        trace_scale = torch.trace(G) / quad_dim
        lambda_reg = alpha * trace_scale + 1e-7
        
        G_reg = G + torch.eye(quad_dim, device=device) * lambda_reg
        rhs = torch.matmul(U_quad_norm.T, N_err_norm)

        # Solve well-conditioned dimensionless system for W_norm
        W_norm = torch.linalg.solve(G_reg, rhs)

        # Exact algebraic un-standardization back to ambient coordinate units:
        # W_true = diag(1 / U_quad_std) @ W_norm @ diag(N_err_std)
        W_i = (W_norm / U_quad_std.T) * N_err_std
        
        print(f"Chart {i:02d} Weingarten Curvature Norm ||W_i||: {W_i.norm().item():.4f}")

        atlas_frames.append({
            'mu': mu_i.cpu(), 
            'Q': Q_i.cpu(), 
            'W': W_i.cpu()
        })

    smooth_sigmas = torch.full((num_charts,), (delta * 0.75)**2, device=device)
    cluster_centers = fps_centers.cpu()

    return membership_mask.cpu(), intrinsic_coords, atlas_frames, cluster_centers, smooth_sigmas, chart_ambient_indices