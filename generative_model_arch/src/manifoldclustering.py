import torch

def construct_whitney_atlas(data: torch.Tensor, 
                            num_charts: int, 
                            intrinsic_dim: int) -> tuple:
    """
    Constructs a true Stéphanovitch / Fefferman Overlapping Submanifold Atlas.
    1. Generates a delta-net via Greedy Farthest Point Sampling (FPS).
    2. Constructs open, overlapping chart neighborhoods B(c_i, 1.5 * delta).
    3. Solves local PCA for the 1st-order tangent frame Q_i.
    4. Regresses the 2nd-order Weingarten curvature tensor W_i.
    """
    N, p = data.shape
    d = intrinsic_dim
    device = data.device

    # ---------------------------------------------------------
    # STEP 1: Greedy Farthest Point Sampling (FPS) Delta-Net
    # Guaranteed to generate maximally separated manifold centers
    # ---------------------------------------------------------
    centroids = torch.zeros(num_charts, p, device=device)
    centroids[0] = data[torch.randint(0, N, (1,))]
    distances = torch.cdist(data, centroids[0].unsqueeze(0)).squeeze(1)

    for i in range(1, num_charts):
        farthest_idx = torch.argmax(distances)
        centroids[i] = data[farthest_idx]
        dist_to_new = torch.cdist(data, centroids[i].unsqueeze(0)).squeeze(1)
        distances = torch.minimum(distances, dist_to_new)

    # The covering radius delta is the maximum distance any data point has to its nearest center
    delta = torch.max(distances).item()
    chart_radius = 1.5 * delta  # 1.5 multiplier mathematically enforces smooth open overlap

    all_pairwise_dists = torch.cdist(data, centroids)
    
    # Boolean mask of overlapping chart memberships (N x m)
    # A single point can (and should) evaluate to True across multiple columns
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

        if N_i < (d + 1) * (d + 2) // 2:
            raise ValueError(f"Chart {i} population ({N_i}) is too sparse to solve 2nd-order Weingarten tensor.")

        # A. Local base center
        mu_i = centroids[i]
        centered_X = X_i - mu_i

        # B. 1st-Order Tangent Plane via PCA
        cov_i = torch.matmul(centered_X.T, centered_X) / (N_i - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_i)
        top_indices = torch.argsort(eigenvalues, descending=True)[:d]
        Q_i = eigenvectors[:, top_indices]  # (p x d)

        # C. 1st-Order Intrinsic Coordinates U_i \in R^{N_i \times d}
        U_i = torch.matmul(centered_X, Q_i)
        intrinsic_coords.append(U_i.cpu())

        # ---------------------------------------------------------
        # STEP 2: 2nd-Order Weingarten Curvature Regression
        # Overcomes the flat beta=2 ceiling by fitting local normal paraboloids
        # ---------------------------------------------------------
        # Calculate normal-bundle error residuals: N_err = X_centered - U_i @ Q_i.T
        N_err = centered_X - torch.matmul(U_i, Q_i.T)  # (N_i x p)
        N_err_std = N_err.std() + 1e-8
        N_err_norm = N_err / N_err_std

        # Generate upper-triangular quadratic outer products of U_i
        # For d=3, generates 6 features: [u1^2, u2^2, u3^2, u1u2, u1u3, u2u3]
        quad_dim = d * (d + 1) // 2
        U_quad = torch.zeros(N_i, quad_dim, device=device)
        col = 0
        for dim1 in range(d):
            for dim2 in range(dim1, d):
                U_quad[:, col] = U_i[:, dim1] * U_i[:, dim2]
                col += 1

        G = torch.matmul(U_quad.T, U_quad)
        
        # Calculate the adaptive scale-invariant penalty \lambda
        alpha = 1e-4
        trace_scale = torch.trace(G) / quad_dim
        lambda_reg = alpha * trace_scale + 1e-7
        
        # Apply isotropic Tikhonov ridge
        G_reg = G + torch.eye(quad_dim, device=device) * lambda_reg
        rhs = torch.matmul(U_quad.T, N_err_norm)

        # Solve the well-conditioned system for W_i \in R^{quad_dim x p}
        W_i = torch.linalg.solve(G_reg, rhs)
        W_i = W_i * N_err_std

        atlas_frames.append({
            'mu': mu_i.cpu(), 
            'Q': Q_i.cpu(), 
            'W': W_i.cpu()
        })

    # Generate soft partition of unity weights evaluated directly on the raw points
    smooth_sigmas = torch.full((num_charts,), (delta * 0.75)**2, device=device)
    cluster_centers = centroids.cpu()

    return membership_mask.cpu(), intrinsic_coords, atlas_frames, cluster_centers, smooth_sigmas, chart_ambient_indices