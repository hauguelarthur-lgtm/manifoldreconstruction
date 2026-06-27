

import torch
import math
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def load_artifacts(data_dir: str):
    """Charge les tenseurs générés par 01_cluster_data.py"""
    print(f"Loading artifacts from {data_dir}...")
    data = torch.load(os.path.join(data_dir, "data.pt"))
    mask = torch.load(os.path.join(data_dir, "membership_mask.pt"))
    coords = torch.load(os.path.join(data_dir, "chart_intrinsic_coords.pt"))
    atlas = torch.load(os.path.join(data_dir, "whitney_atlas.pt"))
    indices = torch.load(os.path.join(data_dir, "chart_ambient_indices.pt"))
    return data, mask, coords, atlas, indices

def vis1_adaptive_delta_net(data, atlas):
    """
    VISUALISATION 1: The Adaptive Delta-Net Covering (Macro-Topology)
    Preuve que le Dual-Condition Radius s'adapte à la densité locale.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=5, c='lightgray', alpha=0.5, label='Ambient Data')
    
    ax = plt.gca()
    for i, chart in enumerate(atlas):
        mu = chart['mu'].numpy()
        r = math.sqrt(chart['r_sq'])
        
        # Plot center
        ax.scatter(mu[0], mu[1], c='red', marker='x', s=50)
        # Plot covering radius
        circle = Circle((mu[0], mu[1]), r, color='blue', fill=False, alpha=0.4, linestyle='--')
        ax.add_patch(circle)
        
    plt.title("Vis 1: Adaptive Delta-Net (Whitney Covering Radii)")
    plt.xlabel("Ambient Dim 1"); plt.ylabel("Ambient Dim 2")
    plt.axis('equal')
    plt.legend()
    plt.show()

def vis2_overlap_multiplicity(data, mask):
    """
    VISUALISATION 2: Overlap Multiplicity Heatmap
    Preuve mathématique de la borne topologique (tau(d)).
    """
    # Sum across the chart dimension to count how many charts claim each point
    multiplicity = mask.sum(dim=1).numpy()
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
                          c=multiplicity, cmap='plasma', s=15, alpha=0.9)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Overlapping Charts (Multiplicity)', rotation=270, labelpad=15)
    
    plt.title("Vis 2: Fefferman Bounded Overlap Property")
    plt.axis('equal')
    plt.show()

def vis3_intrinsic_flattening(coords, chart_idx=0):
    """
    VISUALISATION 3: Intrinsic Tangent Space Flattening (Micro-Topology)
    Montre le dépliage géométrique des données locales.
    """
    U_i = coords[chart_idx].numpy()
    d = U_i.shape[1]
    
    plt.figure(figsize=(8, 6))
    if d == 1:
        # If 1D intrinsic, plot on a line
        plt.scatter(U_i[:, 0], np.zeros_like(U_i[:, 0]), c='green', s=20, alpha=0.7)
        plt.yticks([])
        plt.xlabel("Intrinsic Coordinate u_1")
    elif d >= 2:
        # If 2D+ intrinsic, plot the first two dimensions
        plt.scatter(U_i[:, 0], U_i[:, 1], c='green', s=20, alpha=0.7)
        plt.xlabel("Intrinsic Coordinate u_1"); plt.ylabel("Intrinsic Coordinate u_2")
        
    plt.title(f"Vis 3: Flat Intrinsic Coordinates (Chart {chart_idx})")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def vis4_taylor_curvature_jet(data, atlas, indices, chart_idx=0):
    """
    VISUALISATION 4: Local Taylor Curvature Jets
    Preuve que le tenseur de Weingarten plie le plan tangent pour suivre la variété.
    (Note: Affichage optimisé pour d=1, p=2 pour une visualisation claire)
    """
    chart = atlas[chart_idx]
    idx = indices[chart_idx]
    X_i = data[idx].numpy()
    
    mu = chart['mu'].numpy()
    Q = chart['Q'].numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=5, c='lightgray', alpha=0.3, label='Global Manifold')
    plt.scatter(X_i[:, 0], X_i[:, 1], s=20, c='orange', alpha=0.8, label=f'Local Data (Chart {chart_idx})')
    plt.scatter(mu[0], mu[1], c='red', marker='X', s=100, label='mu_i* (Shifted Apex)')
    
    # Generate continuous curve/surface if W exists and d=1 (for 2D plot)
    if 'W' in chart and Q.shape[1] == 1:
        W = chart['W'].numpy()
        u_grid = np.linspace(-np.max(np.abs(X_i)), np.max(np.abs(X_i)), 100).reshape(-1, 1)
        u_quad = u_grid ** 2  # 2nd order multilinear combination for d=1
        
        # x_proj = mu + Q*u + W*(u^2)
        curve = mu + u_grid @ Q.T + u_quad @ W.T
        plt.plot(curve[:, 0], curve[:, 1], c='blue', linewidth=3, label='Taylor Jet (Weingarten Regression)')
    
    plt.title("Vis 4: High-Order Local Polynomial Jet Reconstruction")
    plt.axis('equal')
    plt.legend()
    plt.show()

def vis5_bump_function_blending(data, atlas, chart_a=0, chart_b=1):
    """
    VISUALISATION 5: Partition of Unity Blending Weights
    Affiche la transition C^infty (fonctions de Fefferman) entre deux cartes adjacentes.
    """
    def eval_bump(x_tensor, chart):
        mu = chart['mu']
        r_sq = chart['r_sq']
        dist_sq = torch.sum((x_tensor - mu) ** 2, dim=1)
        mask = dist_sq < r_sq
        w = torch.zeros_like(dist_sq)
        normalized_sq = dist_sq[mask] / r_sq
        w[mask] = torch.exp(-1.0 / (1.0 - normalized_sq))
        return w

    w_a = eval_bump(data, atlas[chart_a])
    w_b = eval_bump(data, atlas[chart_b])
    
    # Find points that belong to either chart
    active_mask = (w_a > 0) | (w_b > 0)
    X_active = data[active_mask].numpy()
    
    w_a_active = w_a[active_mask].numpy()
    w_b_active = w_b[active_mask].numpy()
    
    # Calculate the normalized transition gradient (0.0 to 1.0)
    relative_weight = w_a_active / (w_a_active + w_b_active + 1e-8)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=5, c='lightgray', alpha=0.3)
    
    scatter = plt.scatter(X_active[:, 0], X_active[:, 1], c=relative_weight, 
                          cmap='coolwarm', s=30, vmin=0, vmax=1)
    
    plt.colorbar(scatter, label=f'Transition Weight (Blue=Chart {chart_b}, Red=Chart {chart_a})')
    plt.scatter(atlas[chart_a]['mu'][0], atlas[chart_a]['mu'][1], c='red', marker='X', s=100)
    plt.scatter(atlas[chart_b]['mu'][0], atlas[chart_b]['mu'][1], c='blue', marker='X', s=100)
    
    plt.title("Vis 5: Fefferman C^infty Mollifier Transition Zone")
    plt.axis('equal')
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