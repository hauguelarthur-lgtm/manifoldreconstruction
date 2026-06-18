import torch
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# ANALYTICAL MANIFOLD GENERATORS (GROUND TRUTH)
# ==========================================

def generate_analytical_sphere(n_samples: int, ambient_dim: int) -> torch.Tensor:
    """
    Generates an exact 2-Sphere (S^2) embedded in R^p.
    Samples uniformly using the cylindrical equal-area projection.
    """
    z = torch.rand(n_samples) * 2 - 1
    phi = torch.rand(n_samples) * 2 * np.pi
    r_xy = torch.sqrt(1 - z**2)
    x = r_xy * torch.cos(phi)
    y = r_xy * torch.sin(phi)
    
    data = torch.zeros(n_samples, ambient_dim)
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z
    return data

def generate_analytical_torus(n_samples: int, ambient_dim: int, R: float = 2.0, r: float = 1.0) -> torch.Tensor:
    """
    Generates an exact 2-Torus (T^2) embedded in R^p.
    Executes rejection sampling to strictly satisfy the Riemannian volume element
    dA = r(R + r cos(theta)) d(theta) d(phi), ensuring a mathematically uniform prior.
    """
    samples = []
    while len(samples) < n_samples:
        theta = torch.rand(n_samples) * 2 * np.pi
        phi = torch.rand(n_samples) * 2 * np.pi
        
        # Maximum density occurs at theta = 0
        p_accept = (R + r * torch.cos(theta)) / (R + r)
        accept = torch.rand(n_samples) < p_accept
        
        valid_theta = theta[accept]
        valid_phi = phi[accept]
        
        for t, p in zip(valid_theta, valid_phi):
            if len(samples) < n_samples:
                x = (R + r * np.cos(t)) * np.cos(p)
                y = (R + r * np.cos(t)) * np.sin(p)
                z = r * np.sin(t)
                samples.append([x, y, z])
                
    tensor_samples = torch.tensor(samples, dtype=torch.float32)
    data = torch.zeros(n_samples, ambient_dim)
    data[:, :3] = tensor_samples
    return data

# ==========================================
# 5.2.1: QUANTITATIVE PAPER METRICS
# ==========================================

def compute_sliced_wasserstein(X: torch.Tensor, Y: torch.Tensor, num_proj: int = 1000) -> float:
    """Computes the 2-Sliced-Wasserstein Distance (SWD)."""
    p = X.size(1)
    projections = torch.randn(p, num_proj, device=X.device)
    projections /= torch.norm(projections, dim=0, keepdim=True)
    
    X_proj, _ = torch.sort(torch.matmul(X, projections), dim=0)
    Y_proj, _ = torch.sort(torch.matmul(Y, projections), dim=0)
    
    if X.size(0) != Y.size(0):
        min_size = min(X.size(0), Y.size(0))
        X_proj = X_proj[torch.randperm(X.size(0))[:min_size].sort()[0]]
        Y_proj = Y_proj[torch.randperm(Y.size(0))[:min_size].sort()[0]]

    return torch.sqrt(torch.mean((X_proj - Y_proj) ** 2)).item()

def compute_mmd_rbf(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> float:
    """Computes Maximum Mean Discrepancy using an RBF kernel."""
    n = min(X.size(0), 5000) # Memory bound for exact pairwise distances
    X_sub = X[torch.randperm(X.size(0))[:n]]
    Y_sub = Y[torch.randperm(Y.size(0))[:n]]

    XX = torch.cdist(X_sub, X_sub, p=2)**2
    YY = torch.cdist(Y_sub, Y_sub, p=2)**2
    XY = torch.cdist(X_sub, Y_sub, p=2)**2
    
    K_XX = torch.exp(-gamma * XX).mean()
    K_YY = torch.exp(-gamma * YY).mean()
    K_XY = torch.exp(-gamma * XY).mean()
    
    mmd_sq = K_XX + K_YY - 2 * K_XY
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0)).item()

def compute_1nn_accuracy(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the 1-Nearest Neighbor Two-Sample Test accuracy.
    Optimal bound is 0.5. Deviations indicate mode collapse or out-of-distribution generation.
    """
    n_samples = min(X.shape[0], Y.shape[0], 5000)
    X_sub = X[np.random.choice(X.shape[0], n_samples, replace=False)]
    Y_sub = Y[np.random.choice(Y.shape[0], n_samples, replace=False)]

    data = np.vstack([X_sub, Y_sub])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, data, labels, cv=5, n_jobs=-1)
    return scores.mean()

# ==========================================
# 5.2.2: QUALITATIVE PAPER VISUALS
# ==========================================

def plot_paper_visuals(target_data: np.ndarray, generated_data: np.ndarray, topology_name: str, output_dir: str):
    """
    Generates publication-grade KDE contour plots and 3D topological scatter projections.
    """
    fig = plt.figure(figsize=(20, 8))
    
    # Execute PCA strictly to isolate the embedding subspace for plotting
    pca_3d = PCA(n_components=3)
    X_pca = pca_3d.fit_transform(target_data)
    Y_pca = pca_3d.transform(generated_data)
    
    # --- 5.2.2.A: 2D Probability Density (KDE) ---
    xmin, xmax = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    ymin, ymax = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    X_grid, Y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    
    ax1 = fig.add_subplot(141)
    kernel_target = gaussian_kde(X_pca[:, :2].T)
    Z_target = np.reshape(kernel_target(positions).T, X_grid.shape)
    ax1.contourf(X_grid, Y_grid, Z_target, levels=15, cmap='Blues')
    ax1.set_title(f"Target Density $\\mu^*$ ({topology_name.capitalize()})", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(142)
    kernel_gen = gaussian_kde(Y_pca[:, :2].T)
    Z_gen = np.reshape(kernel_gen(positions).T, X_grid.shape)
    ax2.contourf(X_grid, Y_grid, Z_gen, levels=15, cmap='Reds')
    ax2.set_title(f"Generated Density $\\hat{{\\mu}}$ ({topology_name.capitalize()})", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # --- 5.2.2.B: 3D Topological Structure ---
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='blue', alpha=0.2, s=5)
    ax3.set_title("Analytical Target Topology", fontsize=12, fontweight='bold')
    ax3.view_init(elev=30., azim=60)

    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(Y_pca[:, 0], Y_pca[:, 1], Y_pca[:, 2], c='red', alpha=0.2, s=5)
    ax4.set_title("Minimax-KSI Generated Topology", fontsize=12, fontweight='bold')
    ax4.view_init(elev=30., azim=60)

    plt.suptitle(f"Section 5.2.2: Minimax-KSI Validation on {topology_name.capitalize()}", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"paper_5_2_2_visuals_{topology_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[VISUAL 5.2.2] Saved qualitative comparison to: {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Computes Section 5.2 metrics against analytical theoretical boundaries.")
    parser.add_argument("--generated_path", type=str, default=os.path.join(project_root, "data", "processed", "generated_samples.pt"))
    parser.add_argument("--topology", type=str, choices=['sphere', 'torus', 'empirical'], default='torus', help="Target topological boundary to evaluate against.")
    parser.add_argument("--empirical_target_path", type=str, default=os.path.join(project_root, "data", "raw", "dataset.pt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "analysis"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.generated_path):
        raise FileNotFoundError(f"Missing generated artifacts at {args.generated_path}.")

    print("Loading generated tensor...")
    generated_data = torch.load(args.generated_path, map_location='cpu')
    ambient_dim = generated_data.shape[1]

    print(f"Generating absolute theoretical boundary for: {args.topology.upper()}")
    if args.topology == 'sphere':
        target_data = generate_analytical_sphere(50000, ambient_dim)
    elif args.topology == 'torus':
        target_data = generate_analytical_torus(50000, ambient_dim)
    else:
        target_data = torch.load(args.empirical_target_path, map_location='cpu')

    print("\n--- SECTION 5.2.1: QUANTITATIVE METRICS ---")
    # Metric execution against the exact continuous measure mu* (approximated via 50k samples)
    swd = compute_sliced_wasserstein(target_data, generated_data)
    mmd = compute_mmd_rbf(target_data, generated_data, gamma=0.1)
    
    target_np = target_data.numpy()
    gen_np = generated_data.numpy()
    nn_acc = compute_1nn_accuracy(target_np, gen_np)
    
    print(f"1. Sliced-Wasserstein Distance (SWD) : {swd:.6f}")
    print(f"2. Maximum Mean Discrepancy (MMD)    : {mmd:.6f}")
    print(f"3. 1-NN Two-Sample Test Accuracy     : {nn_acc:.6f} (Optimal: ~0.50)")
    
    metrics_log = os.path.join(args.output_dir, f"paper_5_2_1_metrics_{args.topology}.txt")
    with open(metrics_log, "w") as f:
        f.write(f"--- SECTION 5.2.1: MINIMAX KSI QUANTITATIVE METRICS ({args.topology.upper()}) ---\n")
        f.write(f"Sliced-Wasserstein Distance : {swd:.6f}\n")
        f.write(f"Maximum Mean Discrepancy    : {mmd:.6f}\n")
        f.write(f"1-NN Two-Sample Accuracy    : {nn_acc:.6f} (Optimal: ~0.50)\n")

    print("\n--- SECTION 5.2.2: QUALITATIVE VISUALS ---")
    plot_paper_visuals(target_np, gen_np, args.topology, args.output_dir)
    print("Verification complete.")

if __name__ == "__main__":
    main()