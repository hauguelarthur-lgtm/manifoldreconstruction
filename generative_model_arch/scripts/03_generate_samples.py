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

from src.wavelet_map import TruncatedBesovWaveletMap
from src.SDEintegrator import generate_samples

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")

    parser = argparse.ArgumentParser(description="Runs the SDE integrator using precomputed eta_t matrices.")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Directory with processed data.")
    parser.add_argument("--ambient_dim", type=int, default=16, help="Ambient space dimension (p).")
    parser.add_argument("--intrinsic_dim", type=int, default=4, help="Intrinsic manifold dimension (d).")
    parser.add_argument("--p_trunc", type=int, default=64, help="Wavelet truncation level (P).")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples to generate.")
    parser.add_argument("--time_steps", type=int, default=50, help="Number of discretization steps matching the solver.")
    args = parser.parse_args()

    # Load artifacts required for SDE generation
    cluster_centers = torch.load(os.path.join(args.data_dir, "cluster_centers.pt"))
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas.pt"))

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc)
    
    print(f"Initializing SDE Integration for {args.num_samples} samples...")
    generated_data = generate_samples(
        model=model,
        precomputed_etas=precomputed_etas,
        cluster_centers=cluster_centers,
        num_samples=args.num_samples,
        ambient_dim=args.ambient_dim,
        num_time_steps=args.time_steps,
        device=torch.device('cpu')
    )

    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(generated_data, output_path)
    print(f"Generation complete. Synthetic manifold samples saved to {output_path}.")

if __name__ == "__main__":
    main()