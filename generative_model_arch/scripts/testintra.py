import torch
import math
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_artifacts(data_dir: str):
    """Charge les tenseurs générés par 01_cluster_data.py"""
    print(f"Loading artifacts from {data_dir}...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    sys.path.insert(0, project_root) if project_root not in sys.path else None
    data = torch.load(os.path.join(data_dir, "data.pt"))
    mask = torch.load(os.path.join(data_dir, "membership_mask.pt"))
    coords = torch.load(os.path.join(data_dir, "chart_intrinsic_coords.pt"))
    atlas_obj = torch.load(os.path.join(data_dir, "whitney_atlas.pt"))
    atlas_frames = atlas_obj.atlas 
    indices = torch.load(os.path.join(data_dir, "chart_ambient_indices.pt"))
    return data, mask, coords, atlas_frames, indices

def enforce_3d_equal_aspect(ax, X, Y, Z):
    """Calcule dynamiquement les limites isométriques pour préserver l'échelle géométrique."""
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def vis1_adaptive_delta_net(data, atlas):
    """
    VISUALISATION 1: The Adaptive Delta-Net Covering (Macro-Topology)
    Projected in R^3 using parametric spherical wireframes[cite: 2].
    """
    np_data = data.numpy()
    X, Y, Z = np_data[:, 0], np_data[:, 1], np_data[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=5, c='lightgray', alpha=0.3, edgecolors='none', label='Ambient Data')
    
    # Pre-compute parametric sphere for covering radius rendering
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 10)
    
    for i, chart in enumerate(atlas):
        mu = chart['mu'].numpy()
        r = math.sqrt(chart['r_sq'])
        
        ax.scatter(mu[0], mu[1], mu[2], c='red', marker='x', s=50)
        
        x_sphere = mu[0] + r * np.outer(np.cos(u), np.sin(v))
        y_sphere = mu[1] + r * np.outer(np.sin(u), np.sin(v))
        z_sphere = mu[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.1)
        
    ax.set_title("Vis 1: Adaptive 3D Delta-Net (Whitney Covering Radii)")
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2"); ax.set_zlabel("x_3")
    enforce_3d_equal_aspect(ax, X, Y, Z)
    plt.legend()
    plt.show()

def vis2_overlap_multiplicity(data, mask):
    """
    VISUALISATION 2: Overlap Multiplicity Heatmap[cite: 2]
    """
    multiplicity = mask.sum(dim=1).numpy()
    np_data = data.numpy()
    X, Y, Z = np_data[:, 0], np_data[:, 1], np_data[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X, Y, Z, c=multiplicity, cmap='plasma', s=15, alpha=0.9, edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Number of Overlapping Charts (Multiplicity)', rotation=270, labelpad=15)
    
    ax.set_title("Vis 2: Fefferman Bounded Overlap Property (3D)")
    enforce_3d_equal_aspect(ax, X, Y, Z)
    plt.show()

def vis3_intrinsic_flattening(coords, chart_idx=0):
    """
    VISUALISATION 3: Intrinsic Tangent Space Flattening (Micro-Topology)[cite: 2]
    """
    U_i = coords[chart_idx].numpy()
    d = U_i.shape[1]
    
    fig = plt.figure(figsize=(8, 6))
    
    if d == 1:
        plt.scatter(U_i[:, 0], np.zeros_like(U_i[:, 0]), c='green', s=20, alpha=0.7)
        plt.yticks([])
        plt.xlabel("Intrinsic Coordinate u_1")
    elif d == 2:
        plt.scatter(U_i[:, 0], U_i[:, 1], c='green', s=20, alpha=0.7)
        plt.xlabel("u_1"); plt.ylabel("u_2")
        plt.axis('equal')
    elif d >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(U_i[:, 0], U_i[:, 1], U_i[:, 2], c='green', s=20, alpha=0.7)
        ax.set_xlabel("u_1"); ax.set_ylabel("u_2"); ax.set_zlabel("u_3")
        enforce_3d_equal_aspect(ax, U_i[:, 0], U_i[:, 1], U_i[:, 2])
        
    plt.title(f"Vis 3: Flat Intrinsic Coordinates (Chart {chart_idx})")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def vis4_taylor_curvature_jet(data, atlas, indices, chart_idx=0):
    """
    VISUALISATION 4: Local Taylor Curvature Jets[cite: 2]
    """
    chart = atlas[chart_idx]
    idx = indices[chart_idx]
    
    np_data = data.numpy()
    X_i = data[idx].numpy()
    X, Y, Z = np_data[:, 0], np_data[:, 1], np_data[:, 2]
    
    mu = chart['mu'].numpy()
    Q = chart['Q'].numpy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X, Y, Z, s=5, c='lightgray', alpha=0.2, edgecolors='none', label='Global Manifold')
    ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], s=20, c='orange', alpha=0.8, edgecolors='none', label=f'Local Data (Chart {chart_idx})')
    ax.scatter(mu[0], mu[1], mu[2], c='red', marker='X', s=100, label='mu_i* (Shifted Apex)')
    
    if 'W' in chart and Q.shape[1] == 1:
        W = chart['W'].numpy()
        u_grid = np.linspace(-np.max(np.abs(X_i)), np.max(np.abs(X_i)), 100).reshape(-1, 1)
        u_quad = u_grid ** 2  
        
        curve = mu + u_grid @ Q.T + u_quad @ W.T
        if curve.shape[1] >= 3:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c='blue', linewidth=3, label='Taylor Jet')
    
    ax.set_title("Vis 4: High-Order Local Polynomial Jet Reconstruction (3D)")
    enforce_3d_equal_aspect(ax, X, Y, Z)
    plt.legend()
    plt.show()

def vis5_bump_function_blending(data, atlas, chart_a=0, chart_b=52):
    """
    VISUALISATION 5: Partition of Unity Blending Weights[cite: 2]
    """
    def eval_bump(x_tensor, chart):
        mu = chart['mu']
        r_sq = chart['r_sq']
        dist_sq = torch.sum((x_tensor - mu) ** 2, dim=1)
        mask = dist_sq < r_sq
        w = torch.zeros_like(dist_sq)
        normalized_sq = dist_sq[mask] / r_sq
        bump_vals = torch.exp(-1.0 / (1.0 - normalized_sq))
        w[mask] = torch.clamp(bump_vals, min=1e-7)
        return w

    w_a = eval_bump(data, atlas[chart_a])
    w_b = eval_bump(data, atlas[chart_b])
    
    active_mask = (w_a > 0) | (w_b > 0)
    X_active = data[active_mask].numpy()
    
    w_a_active = w_a[active_mask].numpy()
    w_b_active = w_b[active_mask].numpy()
    relative_weight = w_a_active / (w_a_active + w_b_active + 1e-8)
    
    np_data = data.numpy()
    X, Y, Z = np_data[:, 0], np_data[:, 1], np_data[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=5, c='lightgray', alpha=0.1, edgecolors='none')
    
    scatter = ax.scatter(X_active[:, 0], X_active[:, 1], X_active[:, 2], 
                         c=relative_weight, cmap='coolwarm', s=30, vmin=0, vmax=1, edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(f'Transition Weight (Blue=Chart {chart_b}, Red=Chart {chart_a})')
    
    mu_a, mu_b = atlas[chart_a]['mu'].numpy(), atlas[chart_b]['mu'].numpy()
    ax.scatter(mu_a[0], mu_a[1], mu_a[2], c='red', marker='X', s=200)
    ax.scatter(mu_b[0], mu_b[1], mu_b[2], c='blue', marker='X', s=200)
    
    ax.set_title("Vis 5: 3D Fefferman $\mathcal{C}^\infty$ Mollifier Transition Zone")
    enforce_3d_equal_aspect(ax, X, Y, Z)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
    args = parser.parse_args()

    data, mask, coords, atlas, indices = load_artifacts(args.data_dir)
    
    print("Generating Visualization 1: Adaptive Delta-Net...")
    vis1_adaptive_delta_net(data, atlas)
    
    print("Generating Visualization 2: Overlap Multiplicity...")
    vis2_overlap_multiplicity(data, mask)
    
    print("Generating Visualization 3: Intrinsic Space...")
    vis3_intrinsic_flattening(coords, chart_idx=0)
    
    print("Generating Visualization 4: Taylor Jet...")
    vis4_taylor_curvature_jet(data, atlas, indices, chart_idx=0)
    
    print("Generating Visualization 5: Partition of Unity Blending...")
    if len(atlas) >= 2:
        vis5_bump_function_blending(data, atlas, chart_a=0, chart_b=min(1, len(atlas)-1))
    else:
        print("Need at least 2 charts for blending visualization.")

if __name__ == "__main__":
    main()