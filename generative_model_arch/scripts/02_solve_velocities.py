# Generates local eta_t coefficients across discrete time steps.
import torch
import os
import sys
import argparse

# Robust path resolution and module import access
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.manifoldclustering import partition_data
from src.wavelet_map import TruncatedBesovWaveletMap

def main():
    default_data_path = os.path.join(project_root, "data", "raw", "dataset.pt")
    default_output_dir = os.path.join(project_root, "data", "processed")

    parser = argparse.ArgumentParser(description="Partitions ambient data into topological patches.")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to raw empirical data tensor (N, p).")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save clustered data.")
    parser.add_argument("--num_charts", type=int, default=10, help="Number of local Euclidean charts (m).")
    args = parser.parse_args()

    # Load processed data
    data = torch.load(os.path.join(args.output_dir, "data.pt"))
    labels = torch.load(os.path.join(args.output_dir, "labels.pt"))
    num_charts = int(labels.max().item() + 1)

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc)
    model.eval()

    time_grid = torch.linspace(0, 1.0, args.time_steps)
    all_etas = []

    print(f"Solving {num_charts} local systems across {args.time_steps} time steps...")

    for step, t in enumerate(time_grid):
        eta_t_local = []
        # Generate matched Gaussian noise for the interpolant targets
        Z = torch.randn_like(data)
        
        # Calculate Stochastic Interpolant components at time t
        # I_t = (1-t)*Z + t*A
        # \dot{I}_t = A - Z
        I_t = (1.0 - t.item()) * Z + t.item() * data
        dot_I_t = data - Z
        
        for i in range(num_charts):
            chart_mask = (labels == i)
            I_t_i = I_t[chart_mask]
            dot_I_t_i = dot_I_t[chart_mask]
            
            if I_t_i.size(0) == 0:
                eta_t_local.append(torch.zeros(args.p_trunc, device=data.device))
                continue
                
            # Compute feature map gradients evaluated at the interpolant state
            feature_grads_i = compute_feature_gradients(model, I_t_i)
            
            # Solve exact linear system K_t \eta_t = r_t
            eta = solve_local_system(feature_grads_i, dot_I_t_i)
            eta_t_local.append(eta)
            
        all_etas.append(eta_t_local)
        if (step + 1) % 10 == 0:
            print(f"Solved step {step + 1}/{args.time_steps}")

    torch.save(all_etas, os.path.join(args.output_dir, "precomputed_etas.pt"))
    print("Linear regression complete. Velocity coefficients saved.")

if __name__ == "__main__":
    main()