import torch
import os
import sys
import argparse
import ot
import yaml
from scipy.spatial.distance import cdist
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.wavelet_map import TruncatedBesovWaveletMap
from src.local_linear_regression import solve_local_system

def main():
    default_config_path = os.path.join(project_root, "configs", "default_config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=default_config_path)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    p_trunc = config['features']['p_trunc']
    time_steps = config['integration']['time_steps']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location='cpu')
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    
    N, m = membership_mask.shape
    d = chart_intrinsic_coords[0].shape[1]

    print(f"Executing Chart-Parallel Exact Optimal Transport strictly in R^{d}...")
    z_clusters_intrinsic = []
    torch.manual_seed(42)

    for i in range(m):
        U_i = chart_intrinsic_coords[i]
        N_i = U_i.shape[0]
        
        if N_i == 0:
            z_clusters_intrinsic.append(torch.zeros((0, d), device=device))
            continue

        # Draw independent standard normal prior strictly for chart i
        Z_raw_i = torch.randn((N_i, d), device=device)
        
        # Evaluate exact squared Euclidean cost matrix on GPU
        cost_matrix_i = torch.cdist(Z_raw_i, U_i, p=2)**2
        
        # Solve Earth Mover's Distance
        plan_i = ot.emd(np.ones(N_i)/N_i, np.ones(N_i)/N_i, cost_matrix_i.cpu().numpy())
        col_ind_i = np.argmax(plan_i, axis=1)
        
        # Formulate exact 1-to-1 sorted intrinsic latents
        Z_sorted_i = Z_raw_i[col_ind_i]
        z_clusters_intrinsic.append(Z_sorted_i)

    torch.save([z.cpu() for z in z_clusters_intrinsic], os.path.join(args.data_dir, "z_clusters_intrinsic.pt"))
    print("Serialized synchronized, non-crossing overlapping latents to z_clusters_intrinsic.pt.")

    # Instantiate and calibrate Besov Wavelet Map
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_trunc).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))
    model.eval()

    time_grid = torch.linspace(0, 1.0 - 1e-5, time_steps)
    all_etas = []

    print(f"Regressing {m} general vector fields (non-irrotational) across {time_steps} steps...")
    for step in range(time_steps):
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
            print(f" Solved step {step + 1}/{time_steps}")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"))
    print("\nPhase 2 & 3 Complete. Vector fields successfully decoupled from potential flow constraint.")

if __name__ == "__main__":
    main()