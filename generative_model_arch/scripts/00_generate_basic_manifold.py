import torch
import os
import argparse
import yaml
import numpy as np

def apply_isometric_embedding(data: torch.Tensor, ambient_dim: int) -> torch.Tensor:
    """
    Entangles a locally-spanned manifold across the full ambient space R^p 
    via a random orthogonal transformation SO(p), strictly preserving intrinsic geodesics.
    """
    # 1. Sample from the standard normal to establish a rotation basis
    H = torch.randn(ambient_dim, ambient_dim)
    
    # 2. Extract orthogonal matrix Q via QR factorization (samples from the Haar measure on O(p))
    Q, R = torch.linalg.qr(H)
    
    # 3. Enforce the Special Orthogonal SO(p) constraint (det(Q) = 1) 
    # This prevents orientation-reversing reflections that alter topological handedness.
    D = torch.diag(torch.sign(torch.diag(R)))
    Q = torch.matmul(Q, D)
    
    # 4. Apply global isometric rotation
    return torch.matmul(data, Q)

def generate_nonlinear_manifold(n_samples: int, intrinsic_dim: int, ambient_dim: int) -> torch.Tensor:
    if ambient_dim < intrinsic_dim * 3:
        raise ValueError("Ambient dimension 'p' must be at least 3x intrinsic dimension 'd'.")
    Z = (torch.rand(n_samples, intrinsic_dim) * 2 - 1) * torch.pi
    X_ambient = torch.zeros(n_samples, ambient_dim)
    for i in range(intrinsic_dim):
        X_ambient[:, i] = Z[:, i]
        X_ambient[:, intrinsic_dim + i] = torch.sin(Z[:, i])
        X_ambient[:, 2 * intrinsic_dim + i] = torch.cos(Z[:, i])
    remaining_dims = ambient_dim - (3 * intrinsic_dim)
    for j in range(remaining_dims):
        idx1 = j % intrinsic_dim
        idx2 = (j + 1) % intrinsic_dim
        X_ambient[:, 3 * intrinsic_dim + j] = Z[:, idx1] * torch.sin(Z[:, idx2])
    return X_ambient

def generate_analytical_sphere(n_samples: int, ambient_dim: int) -> torch.Tensor:
    z = torch.rand(n_samples) * 2 - 1
    phi = torch.rand(n_samples) * 2 * np.pi
    r_xy = torch.sqrt(1 - z**2)
    data = torch.zeros(n_samples, ambient_dim)
    data[:, 0] = r_xy * torch.cos(phi)
    data[:, 1] = r_xy * torch.sin(phi)
    data[:, 2] = z
    return data

def generate_analytical_torus(n_samples: int, ambient_dim: int, R: float = 2.0, r: float = 1.0) -> torch.Tensor:
    samples = []
    collected = 0
    while collected < n_samples:
        # Over-sample batched distributions to account for rejection rate
        guess_n = n_samples - collected
        theta = torch.rand(guess_n * 2) * 2 * torch.pi
        phi = torch.rand(guess_n * 2) * 2 * torch.pi
        
        p_accept = (R + r * torch.cos(theta)) / (R + r)
        accept = torch.rand(guess_n * 2) < p_accept
        
        valid_theta = theta[accept]
        valid_phi = phi[accept]
        
        # Vectorized coordinate computation
        x = (R + r * torch.cos(valid_theta)) * torch.cos(valid_phi)
        y = (R + r * torch.cos(valid_theta)) * torch.sin(valid_phi)
        z = r * torch.sin(valid_theta)
        
        batch = torch.stack([x, y, z], dim=1)
        samples.append(batch)
        collected += batch.shape[0]
        
    data = torch.zeros(n_samples, ambient_dim)
    data[:, :3] = torch.cat(samples, dim=0)[:n_samples]
    return data

def generate_klein_bottle(n_samples: int, ambient_dim: int) -> torch.Tensor:
    if ambient_dim < 4:
        raise ValueError("Klein Bottle embedding requires an ambient dimension 'p' of at least 4.")
    u = torch.rand(n_samples) * 2 * torch.pi
    v = torch.rand(n_samples) * 2 * torch.pi
    R, r = 2.0, 1.0
    data = torch.zeros(n_samples, ambient_dim)
    data[:, 0] = (R + r * torch.cos(v)) * torch.cos(u)
    data[:, 1] = (R + r * torch.cos(v)) * torch.sin(u)
    data[:, 2] = r * torch.sin(v) * torch.cos(u / 2)
    data[:, 3] = r * torch.sin(v) * torch.sin(u / 2)
    return data

def generate_swiss_roll(n_samples: int, ambient_dim: int) -> torch.Tensor:
    if ambient_dim < 3:
        raise ValueError("Swiss Roll embedding requires an ambient dimension 'p' of at least 3.")
    t = (torch.rand(n_samples) * 3 * torch.pi) + 1.5 * torch.pi
    y = torch.rand(n_samples) * 21.0
    data = torch.zeros(n_samples, ambient_dim)
    data[:, 0] = t * torch.cos(t)
    data[:, 1] = y
    data[:, 2] = t * torch.sin(t)
    return data

def generate_n_sphere(n_samples: int, intrinsic_dim: int, ambient_dim: int) -> torch.Tensor:
    if ambient_dim < intrinsic_dim + 1:
        raise ValueError("Ambient dimension 'p' must be at least d + 1 for an S^d sphere.")
    gaussian_samples = torch.randn(n_samples, intrinsic_dim + 1)
    spherical_samples = gaussian_samples / torch.norm(gaussian_samples, dim=1, keepdim=True)
    data = torch.zeros(n_samples, ambient_dim)
    data[:, :intrinsic_dim + 1] = spherical_samples
    return data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "scripts":
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir

    default_config_path = os.path.join(project_root, "configs", "default_config.yaml")
    default_output_dir = os.path.join(project_root, "data", "raw")

    parser = argparse.ArgumentParser(description="Generates synthetic manifold data.")
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--topology", type=str, 
                        choices=['default', 'sphere', 'torus', 'klein', 'swiss_roll', 'n_sphere'], 
                        default='default')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    p = config['manifold']['ambient_dim']
    d = config['manifold']['intrinsic_dim']
    N = config['manifold']['num_samples']

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating empirical dataset: Topology={args.topology.upper()}, N={N}, Ambient Dim={p}")
    
    # 1. Topological mapping
    if args.topology == 'sphere':
        data = generate_analytical_sphere(N, p)
    elif args.topology == 'torus':
        data = generate_analytical_torus(N, p)
    elif args.topology == 'klein':
        data = generate_klein_bottle(N, p)
    elif args.topology == 'swiss_roll':
        data = generate_swiss_roll(N, p)
    elif args.topology == 'n_sphere':
        data = generate_n_sphere(N, d, p)
    else:
        data = generate_nonlinear_manifold(N, d, p)
    
    # 2. Subspace entangling
    if args.topology in ['sphere', 'torus', 'klein', 'swiss_roll', 'n_sphere']:
        print(f"Applying SO({p}) isometric transformation to entangle ambient coordinates...")
        data = apply_isometric_embedding(data, p)
    
    # 3. Serialization
    output_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save(data, output_path)
    print(f"Empirical data tensor saved to {output_path}")

if __name__ == "__main__":
    main()