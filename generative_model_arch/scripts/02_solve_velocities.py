import torch
import os
import sys
import argparse
import ot
from scipy.spatial.distance import cdist
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.wavelet_map import TruncatedBesovWaveletMap
from src.local_linear_regression import solve_local_system

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location='cpu')
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    
    N, m = membership_mask.shape
    d = chart_intrinsic_coords[0].shape[1]

    # -------------------------------------------------------------------------
    # EXACT CORRECTION: Master Global Prior Synchronization (Fixing Violation 3)
    # We match a single global Gaussian prior to the raw data indices once.
    # Shared points across overlapping charts receive the exact same starting latent z_j.
    # -------------------------------------------------------------------------
    print(f"Generating synchronized Master Latent Prior N(0, I_{d}) via Global SVD Proxy matching...")
    torch.manual_seed(42)
    Z_master_raw = torch.randn(N, d, device=device)
    
    # Load raw ambient data to establish global geometric proxy
    data_ambient = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
    centered_ambient = data_ambient - data_ambient.mean(dim=0)
    
    # Extract top 'd' global singular vectors to form U_master_proxy \in R^{N \times d}
    _, _, V = torch.linalg.svd(centered_ambient, full_matrices=False)
    U_master_proxy = torch.matmul(centered_ambient, V[:d].T)
    
    # Solve global Optimal Transport natively in R^d
    cost_matrix = cdist(Z_master_raw.cpu().numpy(), U_master_proxy.cpu().numpy(), metric='sqeuclidean')
    master_plan = ot.emd(np.ones(N)/N, np.ones(N)/N, cost_matrix)
    col_ind = np.argmax(master_plan, axis=1)
    
    Z_master_sorted = torch.zeros_like(Z_master_raw)
    Z_master_sorted[col_ind] = Z_master_raw[np.arange(N)]

    # Slice the globally ordered master latents into the individual overlapping chart streams
    z_clusters_intrinsic = []
    for i in range(m):
        in_chart_idx = torch.nonzero(membership_mask[:, i]).squeeze(1).to(device)
        z_clusters_intrinsic.append(Z_master_sorted[in_chart_idx])

    torch.save([z.cpu() for z in z_clusters_intrinsic], os.path.join(args.data_dir, "z_clusters_intrinsic.pt"))
    print("Serialized synchronized, non-crossing overlapping latents to z_clusters_intrinsic.pt.")

    # Instantiate and calibrate Besov Wavelet Map
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=args.p_trunc).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))
    model.eval()

    time_grid = torch.linspace(0, 1.0 - 1e-5, args.time_steps)
    all_etas = []

    print(f"Regressing {m} general vector fields (non-irrotational) across {args.time_steps} steps...")
    for step in range(args.time_steps):
        t_val = time_grid[step].item()
        eta_t_local = []

        for i in range(m):
            U_i = chart_intrinsic_coords[i]
            Z_i = z_clusters_intrinsic[i]

            I_t_i = (1.0 - t_val) * Z_i + t_val * U_i
            dot_I_t_i = U_i - Z_i

            # Features \phi(x) \in R^{N_i \times P}
            features_i = model(I_t_i)
            
            # Solve for coefficient matrix \eta_i \in R^{P \times d}
            eta_i = solve_local_system(features_i, dot_I_t_i)
            eta_t_local.append(eta_i)

        all_etas.append(eta_t_local)
        if (step + 1) % 25 == 0:
            print(f" Solved step {step + 1}/{args.time_steps}")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"))
    print("\nPhase 2 & 3 Complete. Vector fields successfully decoupled from potential flow constraint.")

if __name__ == "__main__":
    main()