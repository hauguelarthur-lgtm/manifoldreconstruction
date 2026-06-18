import torch
import os
import sys
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--ambient_dim", type=int, default=16)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=64)
    parser.add_argument("--time_steps", type=int, default=50)
    args = parser.parse_args()

    data = torch.load(os.path.join(args.data_dir, "data.pt"))
    labels = torch.load(os.path.join(args.data_dir, "labels.pt"))
    num_charts = int(labels.max().item() + 1)

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc)
    model.eval()

    print("Computing L2 Optimal Transport matching between prior Z and target X...")
    Z_raw = torch.randn_like(data)
    cost_matrix = cdist(Z_raw.cpu().numpy(), data.cpu().numpy(), metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    Z = torch.zeros_like(Z_raw)
    Z[col_ind] = Z_raw[row_ind] 

    z_clusters = [Z[labels == i] for i in range(num_charts)]
    torch.save(z_clusters, os.path.join(args.data_dir, "z_clusters.pt"))

    time_grid = torch.linspace(0, 1.0, args.time_steps)
    all_etas = []
    print(f"Solving {num_charts} local systems across {args.time_steps} time steps...")

    for step, t in enumerate(time_grid):
        eta_t_local = []
        I_t = (1.0 - t.item()) * Z + t.item() * data
        dot_I_t = data - Z
        
        for i in range(num_charts):
            chart_mask = (labels == i)
            I_t_i = I_t[chart_mask]
            dot_I_t_i = dot_I_t[chart_mask]
            
            if I_t_i.size(0) == 0:
                eta_t_local.append(torch.zeros(args.p_trunc, device=data.device))
                continue
                
            feature_grads_i = compute_feature_gradients(model, I_t_i)
            eta = solve_local_system(feature_grads_i, dot_I_t_i)
            eta_t_local.append(eta)
            
        all_etas.append(eta_t_local)
        if (step + 1) % 10 == 0:
            print(f"Solved step {step + 1}/{args.time_steps}")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas.pt"))

if __name__ == "__main__":
    main()