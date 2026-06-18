import torch
import os
import argparse
import yaml

def generate_nonlinear_manifold(num_samples: int, intrinsic_dim: int, ambient_dim: int) -> torch.Tensor:
    """
    Constructs a strictly defined geometric manifold by mapping an intrinsic 
    d-dimensional latent space into a p-dimensional ambient space via non-linear 
    trigonometric and polynomial expansions.
    """
    if ambient_dim < intrinsic_dim * 3:
        raise ValueError("Ambient dimension 'p' must be at least 3x intrinsic dimension 'd' for this specific embedding.")

    # 1. Sample uniformly from the d-dimensional latent hypercube [-pi, pi]^d
    Z = (torch.rand(num_samples, intrinsic_dim) * 2 - 1) * torch.pi
    
    X_ambient = torch.zeros(num_samples, ambient_dim)
    
    # 2. Embed into ambient space using orthogonal non-linear projections
    # This guarantees a complex topology (curvature) that global regressions fail on.
    for i in range(intrinsic_dim):
        X_ambient[:, i] = Z[:, i]                               # Linear component
        X_ambient[:, intrinsic_dim + i] = torch.sin(Z[:, i])    # Curvature component 1
        X_ambient[:, 2 * intrinsic_dim + i] = torch.cos(Z[:, i])# Curvature component 2
        
    # 3. Fill any remaining dimensions with mixed cross-terms to inject structural noise
    remaining_dims = ambient_dim - (3 * intrinsic_dim)
    for j in range(remaining_dims):
        idx1 = j % intrinsic_dim
        idx2 = (j + 1) % intrinsic_dim
        X_ambient[:, 3 * intrinsic_dim + j] = Z[:, idx1] * torch.sin(Z[:, idx2])
        
    return X_ambient

def main():
    parser = argparse.ArgumentParser(description="Generates synthetic manifold data based on YAML config.")
    parser.add_argument("--config", type=str, default="../configs/default_config.yaml", help="Path to config file.")
    parser.add_argument("--output_dir", type=str, default="../data/raw/", help="Output directory for raw tensor.")
    args = parser.parse_args()

    # Load parameters from configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    p = config['manifold']['ambient_dim']
    d = config['manifold']['intrinsic_dim']
    N = config['manifold']['num_samples']

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating synthetic manifold: N={N}, Intrinsic Dim={d}, Ambient Dim={p}")
    data = generate_nonlinear_manifold(num_samples=N, intrinsic_dim=d, ambient_dim=p)
    
    output_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save(data, output_path)
    print(f"Empirical data tensor saved to {output_path}")

if __name__ == "__main__":
    main()