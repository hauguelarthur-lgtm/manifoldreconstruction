import torch
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.wavelet_map import TruncatedBesovWaveletMap, compute_feature_gradients
from src.diffusion import compute_optimal_diffusion
from src.SDEintegrator import generate_samples





def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--ambient_dim", type=int, default=16)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=1024) # Increased feature capacity
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--time_steps", type=int, default=200) # Increased precision
    parser.add_argument("--ode_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = torch.load(os.path.join(args.data_dir, "labels.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas.pt"), map_location=device)
    z_clusters = torch.load(os.path.join(args.data_dir, "z_clusters.pt"), map_location=device)
    cluster_centers = torch.load(os.path.join(args.data_dir, "cluster_centers.pt"), map_location=device)

    num_charts = int(labels.max().item() + 1)
    chart_counts = torch.bincount(labels, minlength=num_charts).float()
    chart_probs = chart_counts / chart_counts.sum()
    chart_assignments = torch.multinomial(chart_probs, args.num_samples, replacement=True).to(device)

    # STRICT CORRECTION 2: Trace numerical stabilizer only (1e-4)
    # Prevents OOD boundary crossings during early integration phase.
    X_0 = torch.zeros(args.num_samples, args.ambient_dim, device=device)
    for i in range(num_charts):
        mask = (chart_assignments == i)
        n_samples_i = mask.sum().item()
        if n_samples_i == 0: continue
        
        Z_i = z_clusters[i].to(device)
        rand_idx = torch.randint(0, Z_i.size(0), (n_samples_i,), device=device)
        X_0[mask] = Z_i[rand_idx] + torch.randn_like(Z_i[rand_idx]) * 1e-4

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc).to(device)
    target_data = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
    model.calibrate(target_data)
    
    mode_str = "ODE (Deterministic)" if args.ode_mode else "SDE (Stochastic)"
    print(f"Initializing {mode_str} Integration for {args.num_samples} samples on {device}...")
    cluster_precisions = torch.load(os.path.join(args.data_dir, "cluster_precisions.pt"), map_location=device)

    generated_data = generate_samples(
        X_0=X_0,
        model=model,
        precomputed_etas=precomputed_etas,
        cluster_centers=cluster_centers,
        cluster_precisions=cluster_precisions,
        num_samples=args.num_samples,
        ambient_dim=args.ambient_dim,
        num_time_steps=args.time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    torch.save(generated_data, os.path.join(args.data_dir, "generated_samples.pt"))

if __name__ == "__main__":
    main()