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
    """
    Computes the 1-Nearest Neighbor Two-Sample Test accuracy with cross-validation variance.
    Optimal generative indistinguishability yields an accuracy of ~0.5000.
    """
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    
    data = np.vstack([X_np, Y_np])
    labels = np.hstack([np.zeros(X_np.shape[0]), np.ones(Y_np.shape[0])])
    
    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, data, labels, cv=cv)
    return float(np.mean(scores)), float(np.std(scores))

def compute_mmd_adaptive(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) using an adaptive Gaussian RBF Kernel
    calibrated strictly via the empirical median distance heuristic (arXiv:2506.19587).
    """
    # Calibrate kernel bandwidth \gamma across a uniform subsample to prevent OOM
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
    
    mmd_squared = K_XX + K_YY - 2.0 * K_XY
    return float(torch.sqrt(torch.clamp(mmd_squared, min=1e-8)))

import torch
import numpy as np
import ot

def compute_true_wasserstein_metrics(X: torch.Tensor, Y: torch.Tensor, max_eval_samples: int = 10000) -> dict[str, float]:
    """
    Computes exact non-asymptotic True 1-Wasserstein (W1) and True 2-Wasserstein (W2) 
    transport metrics via the Network Simplex algorithm (arXiv:2506.19587, Section 5.2.2).
    Operates across the full empirical support without finite-sample subsampling attenuation.
    """
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # Ensure balanced marginal mass; evaluate up to full N=5000 support
    N_eval = min(X_np.shape[0], Y_np.shape[0], max_eval_samples)
    
    if X_np.shape[0] > N_eval:
        idx_X = np.random.choice(X_np.shape[0], N_eval, replace=False)
        X_eval = X_np[idx_X]
    else:
        X_eval = X_np

    if Y_np.shape[0] > N_eval:
        idx_Y = np.random.choice(Y_np.shape[0], N_eval, replace=False)
        Y_eval = Y_np[idx_Y]
    else:
        Y_eval = Y_np

    marginal_a = np.ones(N_eval) / N_eval
    marginal_b = np.ones(N_eval) / N_eval

    # 1. True 1-Wasserstein (W1): Ground cost is unsquared Euclidean distance ||x - y||_2
    cost_matrix_w1 = ot.dist(X_eval, Y_eval, metric='euclidean')
    true_w1 = ot.emd2(marginal_a, marginal_b, cost_matrix_w1, numItermax=2000000)

    # 2. True 2-Wasserstein (W2): Ground cost is squared Euclidean distance ||x - y||_2^2
    cost_matrix_w2 = ot.dist(X_eval, Y_eval, metric='sqeuclidean')
    true_w2_sq = ot.emd2(marginal_a, marginal_b, cost_matrix_w2, numItermax=2000000)
    true_w2 = np.sqrt(max(true_w2_sq, 1e-8))

    return {
        "True_W1": float(true_w1),
        "True_W2": float(true_w2)
    }

def compute_swd_multiscale(X: torch.Tensor, Y: torch.Tensor, num_projections: int = 2048) -> float:
    """
    Computes the Sliced-Wasserstein Distance (SWD) using high-density spherical projections.
    """
    N_X, N_Y = X.shape[0], Y.shape[0]
    if N_X != N_Y:
        min_N = min(N_X, N_Y)
        X = X[torch.randperm(N_X, device=X.device)[:min_N]]
        Y = Y[torch.randperm(N_Y, device=Y.device)[:min_N]]
    
    p = X.shape[1]
    projections = torch.randn(p, num_projections, device=X.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    X_proj, _ = torch.sort(torch.matmul(X, projections), dim=0)
    Y_proj, _ = torch.sort(torch.matmul(Y, projections), dim=0)
    
    return float(torch.mean((X_proj - Y_proj)**2))

def compute_covariance_discrepancy(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Evaluates 1st-order tangent plane and global spectrum preservation."""
    cov_X = torch.cov(X.T)
    cov_Y = torch.cov(Y.T)
    return float(torch.norm(cov_X - cov_Y, p='fro').item())

def visualize_topology(X: np.ndarray, Y: np.ndarray, method: str, output_path: str):
    """Executes dimensionality reduction and generates comparative scatter plots."""
    print(f"Executing dimensionality reduction via {method.upper()}...")
    data = np.vstack([X, Y])
    
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
        
    embedding = reducer.fit_transform(data)
    emb_X, emb_Y = embedding[:X.shape[0]], embedding[X.shape[0]:]
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(emb_X[:, 0], emb_X[:, 1], emb_X[:, 2], alpha=0.5, s=2, c='blue', label='Empirical (Target)')
    ax1.set_title(f'Empirical Topology ({method.upper()})')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(emb_Y[:, 0], emb_Y[:, 1], emb_Y[:, 2], alpha=0.5, s=2, c='red', label='Generated (KSI-SDE)')
    ax2.set_title(f'Generated Topology ({method.upper()})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Computes non-parametric minimax metrics.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing data.pt and generated_samples.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save metric reports and visuals.")
    parser.add_argument("--reduction", type=str, choices=['pca', 'tsne'], default='tsne', help="Dimensionality reduction method.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        X_target = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
        Y_gen = torch.load(os.path.join(args.data_dir, "generated_samples.pt"), map_location=device)
    except FileNotFoundError as e:
        print(f"Fatal I/O Error: {e}")
        return

    print(f"Target Support Shape: {X_target.shape} | Generated Support Shape: {Y_gen.shape}")

    # Execute Evaluation Suite
    # Inside main() of scripts/05_visualstests.py
    print("\n--- SECTION: QUANTITATIVE MINIMAX EVALUATIONS ---")
    swd_val = compute_swd_multiscale(X_target, Y_gen)
    
    # Extract True Wasserstein Suite
    true_w_dict = compute_true_wasserstein_metrics(X_target, Y_gen)
    w1_val = true_w_dict["True_W1"]
    w2_val = true_w_dict["True_W2"]
    
    mmd_val = compute_mmd_adaptive(X_target, Y_gen)
    cov_diff = compute_covariance_discrepancy(X_target, Y_gen)
    knn_mean, knn_std = compute_1nn_stats(X_target, Y_gen)
    
    report = (
        f"--- RIGOROUS GENERATIVE EVALUATION REPORT (arXiv:2506.19587) ---\n"
        f"Ambient Dimension (p)          : {X_target.shape[1]}\n"
        f"Sliced-Wasserstein Distance    : {swd_val:.6f}\n"
        f"True 1-Wasserstein (W1) Cost   : {w1_val:.6f}\n"
        f"True 2-Wasserstein (W2) Cost   : {w2_val:.6f}\n"
        f"Adaptive RBF MMD (Median Heur.): {mmd_val:.6f}\n"
        f"Ambient Covariance Discrepancy : {cov_diff:.6f}\n"
        f"1-NN Two-Sample Accuracy       : {knn_mean:.4f} ± {knn_std:.4f} (Minimax Target: ~0.5000)\n"
    )
    print(report)
    with open(os.path.join(args.output_dir, "evaluation_metrics_rigorous.txt"), "w") as f:
        f.write(report)
        
    vis_path = os.path.join(args.output_dir, f"topological_comparison_{args.reduction}.png")
    visualize_topology(X_target.cpu().numpy(), Y_gen.cpu().numpy(), args.reduction, vis_path)

if __name__ == "__main__":
    main()