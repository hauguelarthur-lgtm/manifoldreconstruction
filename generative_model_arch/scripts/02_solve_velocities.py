import torch
import os
import sys
import argparse
import ot
from scipy.spatial.distance import cdist
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.wavelet_map import TruncatedBesovWaveletMap, compute_feature_gradients
from src.local_linear_regression import solve_local_system

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser(description="Executes Chart-Anchored Intrinsic OT and KSI Drift Regression.")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Phase 1 Intrinsic Whitney Atlas Artifacts
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    num_charts = len(chart_intrinsic_coords)
    d = chart_intrinsic_coords[0].shape[1]

    print(f"Loaded intrinsic coordinates for {num_charts} charts in R^{d} on {device}.")

    # 2. Phase 2: Chart-Anchored Intrinsic Optimal Transport
    print(f"Executing Chart-Parallel Exact Optimal Transport strictly in R^{d}...")
    z_clusters_intrinsic = []

    for i in range(num_charts):
        U_i = chart_intrinsic_coords[i]
        N_i = U_i.shape[0]
        
        if N_i == 0:
            z_clusters_intrinsic.append(torch.zeros((0, d), device=device))
            continue

        # Sample independent Gaussian latent prior strictly in intrinsic dimension d
        Z_raw_i = torch.randn((N_i, d), device=device)
        
        # Uniform marginal distributions
        a = np.ones(N_i) / N_i
        b = np.ones(N_i) / N_i
        
        # Compute cost matrix natively in R^d
        cost_matrix_i = cdist(Z_raw_i.cpu().numpy(), U_i.cpu().numpy(), metric='sqeuclidean')
        
        # Network Simplex converges rapidly and reliably in dense low-dimensional subsets
        T_i = ot.emd(a, b, cost_matrix_i)
        
        col_ind_i = np.argmax(T_i, axis=1)
        row_ind_i = np.arange(N_i)
        
        # Formulate exact 1-to-1 sorted intrinsic latents
        Z_sorted_i = torch.zeros_like(Z_raw_i)
        Z_sorted_i[col_ind_i] = Z_raw_i[row_ind_i]
        
        z_clusters_intrinsic.append(Z_sorted_i)

    torch.save([z.cpu() for z in z_clusters_intrinsic], os.path.join(args.data_dir, "z_clusters_intrinsic.pt"))
    print(f"Serialized sorted intrinsic prior latents to z_clusters_intrinsic.pt.")

    # 3. Phase 3: Intrinsic KSI Drift Regression
    # Instantiate Truncated Besov Wavelet Map natively in intrinsic dimension d
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=args.p_trunc).to(device)
    
    # Calibrate bandwidths over the concatenated intrinsic manifold support
    U_all = torch.cat(chart_intrinsic_coords, dim=0)
    print("Calibrating Besov Wavelet bandwidths via median pairwise distance of intrinsic coordinates...")
    model.calibrate(U_all)
    model.eval()

    # Synchronized temporal grid boundary
    time_grid = torch.linspace(0, 1.0 - 1e-5, args.time_steps)
    print(f"Solving {num_charts} chart-anchored Besov vector fields across {args.time_steps} time steps...")

    all_etas = []

    for step in range(args.time_steps):
        t_val = time_grid[step].item()
        eta_t_local = []

        for i in range(num_charts):
            U_i = chart_intrinsic_coords[i]
            Z_i = z_clusters_intrinsic[i]
            N_i = U_i.shape[0]

            if N_i == 0:
                eta_t_local.append(torch.zeros(args.p_trunc, device=device))
                continue

            # Formulate intrinsic interpolant streamlines and target probability flow velocities
            I_t_i = (1.0 - t_val) * Z_i + t_val * U_i
            dot_I_t_i = U_i - Z_i

            # Evaluate Besov feature gradients strictly in R^d
            feature_grads_i = compute_feature_gradients(model, I_t_i)
            
            # Regress wavelet coefficients via trace-scaled Tikhonov stabilization
            eta_i = solve_local_system(feature_grads_i, dot_I_t_i)
            eta_t_local.append(eta_i)

        all_etas.append(eta_t_local)
        if (step + 1) % 20 == 0 or step == args.time_steps - 1:
            print(f"Solved step {step + 1}/{args.time_steps} (t = {t_val:.4f})")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"))
    print(f"\nPhase 2 & Phase 3 Complete. Serialized velocity coefficients to precomputed_etas_intrinsic.pt.")

if __name__ == "__main__":
    main()