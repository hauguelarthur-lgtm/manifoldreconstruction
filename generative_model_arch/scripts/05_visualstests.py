import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import ot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_1nn_stats(X: torch.Tensor, Y: torch.Tensor, cv: int = 5) -> tuple[float, float]:
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    data = np.vstack([X_np, Y_np])
    labels = np.hstack([np.zeros(X_np.shape[0]), np.ones(Y_np.shape[0])])
    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, data, labels, cv=cv)
    return float(np.mean(scores)), float(np.std(scores))

def compute_mmd_adaptive(X: torch.Tensor, Y: torch.Tensor) -> float:
    N_sub = min(X.shape[0], 2000)
    idx_sub = torch.randperm(X.shape[0], device=X.device)[:N_sub]
    pairwise_sq_dists = torch.pdist(X[idx_sub], p=2)**2
    median_dist_sq = torch.median(pairwise_sq_dists).clamp(min=1e-4)
    gamma = float(1.0 / (2.0 * median_dist_sq.item()))

    XX = torch.cdist(X, X, p=2)**2
    YY = torch.cdist(Y, Y, p=2)**2
    XY = torch.cdist(X, Y, p=2)**2
    
    K_XX = torch.exp(-gamma * XX).mean()
    K_YY = torch.exp(-gamma * YY).mean()
    K_XY = torch.exp(-gamma * XY).mean()
    return float(torch.sqrt(torch.clamp(K_XX + K_YY - 2.0 * K_XY, min=1e-8)))

def compute_true_wasserstein_metrics(X: torch.Tensor, Y: torch.Tensor, max_eval_samples: int = 10000) -> dict[str, float]:
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    N_eval = min(X_np.shape[0], Y_np.shape[0], max_eval_samples)
    
    X_eval = X_np[np.random.choice(X_np.shape[0], N_eval, replace=False)] if X_np.shape[0] > N_eval else X_np
    Y_eval = Y_np[np.random.choice(Y_np.shape[0], N_eval, replace=False)] if Y_np.shape[0] > N_eval else Y_np

    marginal_a = np.ones(N_eval) / N_eval
    marginal_b = np.ones(N_eval) / N_eval

    cost_w1 = ot.dist(X_eval, Y_eval, metric='euclidean')
    true_w1 = ot.emd2(marginal_a, marginal_b, cost_w1, numItermax=2000000)

    cost_w2 = ot.dist(X_eval, Y_eval, metric='sqeuclidean')
    true_w2 = np.sqrt(max(ot.emd2(marginal_a, marginal_b, cost_w2, numItermax=2000000), 1e-8))
    return {"True_W1": float(true_w1), "True_W2": float(true_w2)}

def compute_swd_multiscale(X: torch.Tensor, Y: torch.Tensor, num_projections: int = 2048) -> float:
    """
    MATHEMATICAL AMELIORATION: Evaluates exact 1D Optimal Transport across projected slices
    between unequal empirical measures via continuous quantile interpolation (Zero Subsampling Loss).
    """
    p = X.shape[1]
    projections = torch.randn(p, num_projections, device=X.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    X_proj, _ = torch.sort(torch.matmul(X, projections), dim=0) # Shape: (N_X, num_proj)
    Y_proj, _ = torch.sort(torch.matmul(Y, projections), dim=0) # Shape: (N_Y, num_proj)
    
    N_X, N_Y = X.shape[0], Y.shape[0]
    if N_X == N_Y:
        return float(torch.mean((X_proj - Y_proj)**2))
        
    # Generalized 1D Inverse-CDF Quantile Integration
    u_grid = torch.linspace(0, 1, steps=max(N_X, N_Y), device=X.device)
    
    def interpolate_quantile(sorted_proj, N_orig):
        orig_grid = torch.linspace(0, 1, steps=N_orig, device=X.device)
        return torch.stack([torch.tensor(np.interp(u_grid.cpu().numpy(), orig_grid.cpu().numpy(), sorted_proj[:, j].cpu().numpy()), device=X.device) for j in range(num_projections)], dim=1)

    X_interp = interpolate_quantile(X_proj, N_X)
    Y_interp = interpolate_quantile(Y_proj, N_Y)
    return float(torch.mean((X_interp - Y_interp)**2))

def compute_covariance_discrepancy(X: torch.Tensor, Y: torch.Tensor) -> float:
    """MATHEMATICAL AMELIORATION: Computes Relative Spectral Frobenius Discrepancy."""
    cov_X = torch.cov(X.T)
    cov_Y = torch.cov(Y.T)
    norm_X = torch.norm(cov_X, p='fro')
    return float((torch.norm(cov_X - cov_Y, p='fro') / (norm_X + 1e-6)).item())

def visualize_topology(X: np.ndarray, Y: np.ndarray, method: str, output_path: str):
    """
    RIGOROUS MATHEMATICAL REWORK (PCA & TOPOLOGICAL RENDERING):
    1. Canonical Geometry Preservation: Bypasses PCA rotation if p <= 3 to preserve physical axes.
    2. Joint Superimposition: Overlays Target (blue) and Generated (red) in a unified 3D frame.
    3. Multi-Planar Orthographic Shadows: Generates XY, XZ, and YZ 2D projections to unmask hidden voids.
    4. Spectral Accounting: Computes and displays the Cumulative Explained Variance Ratio (EVR).
    """
    p = X.shape[1]
    data = np.vstack([X, Y])
    
    # --- SPECTRAL PROJECTION & ENERGY ACCOUNTING ---
    if method.lower() == 'pca':
        if p <= 3:
            # Strictly preserve canonical physical geometry if ambient space is R^2 or R^3
            padded_data = np.zeros((data.shape[0], 3))
            padded_data[:, :p] = data
            emb_X = padded_data[:X.shape[0]]
            emb_Y = padded_data[X.shape[0]:]
            evr_str = f"Canonical Physical Geometry (p={p})"
        else:
            reducer = PCA(n_components=3)
            embedding = reducer.fit_transform(data)
            emb_X = embedding[:X.shape[0]]
            emb_Y = embedding[X.shape[0]:]
            evr_total = float(np.sum(reducer.explained_variance_ratio_) * 100.0)
            evr_str = f"PCA Top 3 Subspace (EVR: {evr_total:.2f}%)"
    else:
        # t-SNE Fallback
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
        embedding = reducer.fit_transform(data)
        emb_X = embedding[:X.shape[0]]
        emb_Y = embedding[X.shape[0]:]
        evr_str = "t-SNE Non-Linear Manifold Embedding"

    # --- 4-PANEL DIAGNOSTIC RENDERING ENGINE ---
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Joint 3D Superimposition (Perspective Axis)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(emb_X[:, 0], emb_X[:, 1], emb_X[:, 2], alpha=0.35, s=4, c='#1f77b4', label='Empirical Ground Truth (X)')
    ax1.scatter(emb_Y[:, 0], emb_Y[:, 1], emb_Y[:, 2], alpha=0.45, s=3, c='#d62728', label='Generated KSI Flow (Y)')
    ax1.set_title(f"Joint 3D Superimposition\n[{evr_str}]", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Principal Axis 1"); ax1.set_ylabel("Principal Axis 2"); ax1.set_zlabel("Principal Axis 3")
    ax1.legend(loc='upper right', markerscale=3.0)
    ax1.view_init(elev=25.0, azim=-55.0)

    # Panel 2: XY Planar Orthographic Projection (Top View)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(emb_X[:, 0], emb_X[:, 1], alpha=0.30, s=3, c='#1f77b4')
    ax2.scatter(emb_Y[:, 0], emb_Y[:, 1], alpha=0.40, s=2, c='#d62728')
    ax2.set_title("Orthographic Shadow: XY (Top View)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Principal Axis 1"); ax2.set_ylabel("Principal Axis 2")
    ax2.grid(True, linestyle='--', alpha=0.5); ax2.set_aspect('equal', 'datalim')

    # Panel 3: XZ Planar Orthographic Projection (Front View)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(emb_X[:, 0], emb_X[:, 2], alpha=0.30, s=3, c='#1f77b4')
    ax3.scatter(emb_Y[:, 0], emb_Y[:, 2], alpha=0.40, s=2, c='#d62728')
    ax3.set_title("Orthographic Shadow: XZ (Front View)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Principal Axis 1"); ax3.set_ylabel("Principal Axis 3")
    ax3.grid(True, linestyle='--', alpha=0.5); ax3.set_aspect('equal', 'datalim')

    # Panel 4: YZ Planar Orthographic Projection (Side View)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(emb_X[:, 1], emb_X[:, 2], alpha=0.30, s=3, c='#1f77b4')
    ax4.scatter(emb_Y[:, 1], emb_Y[:, 2], alpha=0.40, s=2, c='#d62728')
    ax4.set_title("Orthographic Shadow: YZ (Side View)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Principal Axis 2"); ax4.set_ylabel("Principal Axis 3")
    ax4.grid(True, linestyle='--', alpha=0.5); ax4.set_aspect('equal', 'datalim')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization Engine] Reworked 4-panel diagnostic dashboard rendered -> {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reduction", type=str, choices=['pca', 'tsne'], default='tsne')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    X_target = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
    Y_gen = torch.load(os.path.join(args.data_dir, "generated_samples.pt"), map_location=device)

    print("\n--- SECTION: QUANTITATIVE MINIMAX EVALUATIONS ---")
    swd_val = compute_swd_multiscale(X_target, Y_gen)
    true_w = compute_true_wasserstein_metrics(X_target, Y_gen)
    mmd_val = compute_mmd_adaptive(X_target, Y_gen)
    cov_diff = compute_covariance_discrepancy(X_target, Y_gen)
    knn_mean, knn_std = compute_1nn_stats(X_target, Y_gen)
    
    report = (
        f"--- RIGOROUS GENERATIVE EVALUATION REPORT (arXiv:2506.19587) ---\n"
        f"Ambient Dimension (p)          : {X_target.shape[1]}\n"
        f"Sliced-Wasserstein Distance    : {swd_val:.6f}\n"
        f"True 1-Wasserstein (W1) Cost   : {true_w['True_W1']:.6f}\n"
        f"True 2-Wasserstein (W2) Cost   : {true_w['True_W2']:.6f}\n"
        f"Adaptive RBF MMD (Median Heur.): {mmd_val:.6f}\n"
        f"Relative Covariance Discrepancy: {cov_diff:.6f} (Normalized)\n"
        f"1-NN Two-Sample Accuracy       : {knn_mean:.4f} ± {knn_std:.4f} (Minimax Target: ~0.5000)\n"
    )
    print(report)
    with open(os.path.join(args.output_dir, "evaluation_metrics_rigorous.txt"), "w") as f: f.write(report)
    visualize_topology(X_target.cpu().numpy(), Y_gen.cpu().numpy(), args.reduction, os.path.join(args.output_dir, f"topological_comparison_{args.reduction}.png"))

if __name__ == "__main__": main()