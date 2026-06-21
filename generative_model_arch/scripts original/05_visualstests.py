import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_1nn_accuracy(X: torch.Tensor, Y: torch.Tensor, cv: int = 5) -> float:
    """
    Computes the 1-Nearest Neighbor Two-Sample Test accuracy.
    Optimal generative indistinguishability yields an accuracy of ~0.50.
    """
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    data = np.vstack([X_np, Y_np])
    labels = np.hstack([np.zeros(X_np.shape[0]), np.ones(Y_np.shape[0])])
    
    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, data, labels, cv=cv)
    return float(np.mean(scores))

def compute_mmd(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> float:
    """
    Computes the Maximum Mean Discrepancy (MMD) using a Gaussian RBF Kernel.
    Operates in O(N^2) utilizing batched PyTorch matrix operations.
    """
    XX = torch.cdist(X, X, p=2)**2
    YY = torch.cdist(Y, Y, p=2)**2
    XY = torch.cdist(X, Y, p=2)**2
    
    K_XX = torch.exp(-gamma * XX).mean()
    K_YY = torch.exp(-gamma * YY).mean()
    K_XY = torch.exp(-gamma * XY).mean()
    
    mmd_squared = K_XX + K_YY - 2 * K_XY
    return float(torch.sqrt(torch.clamp(mmd_squared, min=1e-8)))

def compute_swd(X: torch.Tensor, Y: torch.Tensor, num_projections: int = 1000) -> float:
    """
    Computes the Sliced-Wasserstein Distance (SWD).
    Projects p-dimensional data onto random 1D hyper-spherical vectors.
    """
    N_X, N_Y = X.shape[0], Y.shape[0]
    if N_X != N_Y:
        min_N = min(N_X, N_Y)
        X = X[torch.randperm(N_X, device=X.device)[:min_N]]
        Y = Y[torch.randperm(N_Y, device=Y.device)[:min_N]]
    # ------------------------------
    
    p = X.shape[1]
    
    # Generate random projection vectors uniformly on the unit sphere S^{p-1}
    projections = torch.randn(p, num_projections, device=X.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    # Project data
    X_proj = torch.matmul(X, projections)
    Y_proj = torch.matmul(Y, projections)
    
    # Sort projections to compute the exact 1D Wasserstein distance
    X_proj_sorted, _ = torch.sort(X_proj, dim=0)
    Y_proj_sorted, _ = torch.sort(Y_proj, dim=0)
    
    # Compute L2 distance between sorted projections
    wasserstein_distances = torch.mean((X_proj_sorted - Y_proj_sorted)**2, dim=0)
    
    return float(torch.mean(wasserstein_distances))

def visualize_topology(X: np.ndarray, Y: np.ndarray, method: str, output_path: str):
    """
    Executes dimensionality reduction and generates comparative scatter visualizations.
    """
    print(f"Executing dimensionality reduction via {method.upper()}...")
    data = np.vstack([X, Y])
    
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
        
    embedding = reducer.fit_transform(data)
    
    emb_X = embedding[:X.shape[0]]
    emb_Y = embedding[X.shape[0]:]
    
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
    parser = argparse.ArgumentParser(description="Computes minimax metrics and visualizes high-dimensional topologies.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing data.pt and generated_samples.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save metric reports and visuals.")
    parser.add_argument("--reduction", type=str, choices=['pca', 'tsne'], default='tsne', help="Dimensionality reduction method for visualization.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tensors
    try:
        X_target = torch.load(os.path.join(args.data_dir, "data.pt")).to(device)
        Y_gen = torch.load(os.path.join(args.data_dir, "generated_samples.pt")).to(device)
    except FileNotFoundError as e:
        print(f"I/O Error: {e}")
        return

    print(f"Target Support Shape: {X_target.shape} | Generated Support Shape: {Y_gen.shape}")

    # 1. Compute Quantitative Minimax Metrics
    print("\n--- SECTION: QUANTITATIVE METRICS ---")
    swd_val = compute_swd(X_target, Y_gen)
    mmd_val = compute_mmd(X_target, Y_gen)
    knn_val = compute_1nn_accuracy(X_target, Y_gen)
    
    report = (
        f"--- TOPOLOGICAL GENERATION METRICS ---\n"
        f"Ambient Dimension (p)       : {X_target.shape[1]}\n"
        f"Sliced-Wasserstein Distance : {swd_val:.6f}\n"
        f"Maximum Mean Discrepancy    : {mmd_val:.6f}\n"
        f"1-NN Two-Sample Accuracy    : {knn_val:.6f} (Optimal: ~0.50)\n"
    )
    
    print(report)
    with open(os.path.join(args.output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(report)
        
    # 2. Compute Topological Visualization
    vis_path = os.path.join(args.output_dir, f"manifold_comparison_{args.reduction}.png")
    visualize_topology(X_target.cpu().numpy(), Y_gen.cpu().numpy(), args.reduction, vis_path)

if __name__ == "__main__":
    main()