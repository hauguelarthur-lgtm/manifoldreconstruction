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

from src.wavelet_map import TruncatedBesovWaveletMap
from src.SDEintegrator import generate_samples
from src.projector import GlobalSubspaceProjector

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--ode_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Algebraic Projector and Reduced State Tensors
    projector = torch.load(os.path.join(args.data_dir, "projector.pt"), map_location=device)
    k = projector.k
    
    labels = torch.load(os.path.join(args.data_dir, "labels.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas.pt"), map_location=device)
    z_clusters = torch.load(os.path.join(args.data_dir, "z_clusters.pt"), map_location=device)
    
    # Load R^k specific topological artifacts
    cluster_centers_k = torch.load(os.path.join(args.data_dir, "cluster_centers_k.pt"), map_location=device)
    cluster_precisions_k = torch.load(os.path.join(args.data_dir, "cluster_precisions_k.pt"), map_location=device)
    
    data_k = torch.load(os.path.join(args.data_dir, "data_k.pt"), map_location=device)
    num_charts = int(labels.max().item() + 1)
    
    # Extract the distinct initial and terminal barycenters
    centers_1 = torch.stack([data_k[labels == i].mean(dim=0) for i in range(num_charts)]).to(device)
    centers_0 = torch.stack([z_clusters[i].mean(dim=0) for i in range(num_charts)]).to(device)

    chart_counts = torch.bincount(labels, minlength=num_charts).float()
    chart_probs = chart_counts / chart_counts.sum()
    chart_assignments = torch.multinomial(chart_probs, args.num_samples, replacement=True).to(device)

    X_0 = torch.zeros(args.num_samples, k, device=device)
    for i in range(num_charts):
        mask = (chart_assignments == i)
        n_samples_i = mask.sum().item()
        if n_samples_i == 0: continue
        
        Z_i = z_clusters[i].to(device)
        rand_idx = torch.randint(0, Z_i.size(0), (n_samples_i,), device=device)
        # Initialize directly on the discrete optimal transport prior
        X_0[mask] = Z_i[rand_idx] 

    model = TruncatedBesovWaveletMap(k, args.intrinsic_dim, args.p_trunc).to(device)
    model.calibrate(data_k)
    
    mode_str = "ODE (Deterministic)" if args.ode_mode else "SDE (Stochastic)"
    print(f"Initializing {mode_str} Integration for {args.num_samples} samples in R^{k} on {device}...")
    
    generated_data_k = generate_samples(
        X_0=X_0,
        model=model,
        precomputed_etas=precomputed_etas,
        centers_0=centers_0,
        centers_1=centers_1,
        cluster_precisions=cluster_precisions_k,
        num_samples=args.num_samples,
        ambient_dim=k,
        num_time_steps=args.time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    print(f"Lifting generated coordinates from R^{k} to ambient R^{projector.mean.shape[1]}...")
    generated_data = projector.inverse_transform(generated_data_k.cpu()).to(device)

    torch.save(generated_data, os.path.join(args.data_dir, "generated_samples.pt"))

if __name__ == "__main__":
    main()