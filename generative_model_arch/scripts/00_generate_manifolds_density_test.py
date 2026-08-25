import torch
import math
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

# =====================================================================
# Intrinsic Measure Samplers
# =====================================================================

def sample_intrinsic_parameter(n_samples: int, bounds: tuple, density_type: str = 'uniform') -> torch.Tensor:
    """
    Samples intrinsic coordinates strictly within the specified geometric bounds
    while enforcing the selected non-linear probability density function.
    """
    low, high = bounds
    span = high - low
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if density_type == 'uniform':
        return torch.rand(n_samples, device=device) * span + low
        
    elif density_type == 'multimodal':
        # Gaussian mixture strictly truncated to the defined bounds
        modes = torch.tensor([low + 0.25 * span, low + 0.75 * span], device=device)
        scales = torch.tensor([0.05 * span, 0.05 * span], device=device)
        samples = torch.empty(n_samples, device=device)
        
        assigned = 0
        while assigned < n_samples:
            batch = 2000
            assignments = torch.randint(0, 2, (batch,), device=device)
            cand = modes[assignments] + scales[assignments] * torch.randn(batch, device=device)
            valid_mask = (cand >= low) & (cand <= high)
            valid = cand[valid_mask]
            
            take = min(valid.size(0), n_samples - assigned)
            samples[assigned:assigned+take] = valid[:take]
            assigned += take
        return samples
        
    elif density_type == 'exponential':
        # Inverse transform sampling for truncated exponential decay
        lam = 5.0
        u = torch.rand(n_samples, device=device)
        norm_samples = - (1.0 / lam) * torch.log(1.0 - u * (1.0 - math.exp(-lam)))
        return norm_samples * span + low
        
    elif density_type == 'oscillatory':
        # Rejection sampling evaluating p(u) \propto \sin^2(k\pi u)
        samples = torch.empty(0, device=device)
        while samples.size(0) < n_samples:
            u_cand = torch.rand(n_samples, device=device)
            p_eval = torch.sin(4.0 * math.pi * u_cand)**2
            accept_mask = torch.rand(n_samples, device=device) < p_eval
            samples = torch.cat([samples, u_cand[accept_mask]])
        return samples[:n_samples] * span + low
        
    else:
        raise ValueError(f"Undefined density_type: {density_type}")

def sample_spherical_measure(n_samples: int, ambient_dim: int, noise: float, density_type: str) -> torch.Tensor:
    """
    Executes rejection sampling directly on the S^{d} hypersphere surface
    to evaluate anisotropic probability mass concentrations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = torch.empty((0, ambient_dim), device=device)
    
    while samples.size(0) < n_samples:
        data = torch.randn(n_samples, ambient_dim, device=device)
        data = data / torch.norm(data, dim=1, keepdim=True)
        
        if density_type == 'uniform':
            accept_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
        elif density_type == 'multimodal':
            # Concentrate mass at the poles of the first dimension
            p_eval = torch.exp(2.0 * (torch.abs(data[:, 0]) - 1.0))
            accept_mask = torch.rand(n_samples, device=device) < p_eval
        elif density_type == 'oscillatory':
            # Latitudinal density oscillations
            p_eval = torch.sin(3.0 * math.pi * data[:, 0])**2
            accept_mask = torch.rand(n_samples, device=device) < p_eval
        elif density_type == 'exponential':
            # Mass concentration strictly at the positive pole
            p_eval = torch.exp(3.0 * (data[:, 0] - 1.0))
            accept_mask = torch.rand(n_samples, device=device) < p_eval
            
        samples = torch.cat([samples, data[accept_mask]], dim=0)
        
    data = samples[:n_samples]
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

# =====================================================================
# 1D Manifolds (d=1)
# =====================================================================

def generate_simple_1d_helix(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    t = sample_intrinsic_parameter(n_samples, (0.0, 10 * math.pi), density)
    x = torch.cos(t)
    y = torch.sin(t)
    z = 0.1 * t
    data = torch.stack([x, y, z], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_1d_trefoil(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    t = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    x = torch.sin(t) + 2 * torch.sin(2 * t)
    y = torch.cos(t) - 2 * torch.cos(2 * t)
    z = -torch.sin(3 * t)
    data = torch.stack([x, y, z], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_1d_lissajous(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    t = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    x = torch.sin(3 * t)
    y = torch.sin(4 * t)
    z = torch.cos(5 * t)
    data = torch.stack([x, y, z], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

# =====================================================================
# 2D Manifolds (d=2)
# =====================================================================

def generate_simple_2d_sphere(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    return sample_spherical_measure(n_samples, ambient_dim=3, noise=noise, density_type=density)

def generate_complex_2d_swiss_roll(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    u = sample_intrinsic_parameter(n_samples, (0.0, 1.0), density)
    t = 1.5 * math.pi * (1 + 2 * u)
    y = sample_intrinsic_parameter(n_samples, (0.0, 21.0), density)
    x = t * torch.cos(t)
    z = t * torch.sin(t)
    data = torch.stack([x, y, z], dim=1)
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_hyperbolic_saddle(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    x = sample_intrinsic_parameter(n_samples, (-2.0, 2.0), density)
    y = sample_intrinsic_parameter(n_samples, (-2.0, 2.0), density)
    z = x**2 - y**2
    data = torch.stack([x, y, z], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_mobius_strip(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    u = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    v = sample_intrinsic_parameter(n_samples, (-0.5, 0.5), density)
    x = (1 + v * torch.cos(u / 2)) * torch.cos(u)
    y = (1 + v * torch.cos(u / 2)) * torch.sin(u)
    z = v * torch.sin(u / 2)
    data = torch.stack([x, y, z], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_klein_bottle(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    u = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    v = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    r1, r2 = 2.0, 1.0
    x1 = (r1 + r2 * torch.cos(v)) * torch.cos(u)
    x2 = (r1 + r2 * torch.cos(v)) * torch.sin(u)
    x3 = r2 * torch.sin(v) * torch.cos(u / 2)
    x4 = r2 * torch.sin(v) * torch.sin(u / 2)
    data = torch.stack([x1, x2, x3, x4], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

# =====================================================================
# High-Dimensional Manifolds (d > 2)
# =====================================================================

def generate_simple_d3_hypersphere(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    return sample_spherical_measure(n_samples, ambient_dim=4, noise=noise, density_type=density)

def generate_complex_d3_torus_product(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    u = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    v = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    w = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    R, r, r2 = 2.0, 0.5, 1.0
    x1 = (R + r * torch.cos(v)) * torch.cos(u)
    x2 = (R + r * torch.cos(v)) * torch.sin(u)
    x3 = r * torch.sin(v)
    x4 = r2 * torch.cos(w)
    x5 = r2 * torch.sin(w)
    data = torch.stack([x1, x2, x3, x4, x5], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

def generate_complex_d3_flat_torus(n_samples: int, noise: float = 0.0, density: str = 'uniform') -> torch.Tensor:
    u1 = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    u2 = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    u3 = sample_intrinsic_parameter(n_samples, (0.0, 2 * math.pi), density)
    x1, x2 = torch.cos(u1), torch.sin(u1)
    x3, x4 = torch.cos(u2), torch.sin(u2)
    x5, x6 = torch.cos(u3), torch.sin(u3)
    data = torch.stack([x1, x2, x3, x4, x5, x6], dim=1)
    if noise > 0: data += torch.randn_like(data) * noise
    return data

# =====================================================================
# Execution
# =====================================================================

def main():
    generators = {
        "simple_1d_helix": generate_simple_1d_helix,
        "complex_1d_trefoil": generate_complex_1d_trefoil,
        "complex_1d_lissajous": generate_complex_1d_lissajous,
        "simple_2d_sphere": generate_simple_2d_sphere,
        "complex_2d_swiss_roll": generate_complex_2d_swiss_roll,
        "complex_2d_saddle": generate_complex_2d_hyperbolic_saddle,
        "complex_2d_mobius": generate_complex_2d_mobius_strip,
        "complex_2d_klein": generate_complex_2d_klein_bottle,
        "simple_d3_hypersphere": generate_simple_d3_hypersphere,
        "complex_d3_torus_product": generate_complex_d3_torus_product,
        "complex_d3_flat_torus": generate_complex_d3_flat_torus
    }

    parser = argparse.ArgumentParser(description="Generate synthetic manifolds for generative geometry testing.")
    parser.add_argument("--manifold", type=str, required=True, 
                        choices=list(generators.keys()),
                        help="The topological structure to generate.")
    parser.add_argument("--density", type=str, default='uniform',
                        choices=['uniform', 'multimodal', 'exponential', 'oscillatory'],
                        help="Intrinsic probability measure distribution.")
    parser.add_argument("--out_dir", type=str, default="../generative_model_arch/data/raw", help="Output directory.")
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    num_samples = int(config['manifold']['num_samples'])
    noise = float(config['manifold']['noise'])

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Generating {args.manifold} (N={num_samples}, noise={noise}, density={args.density})...")
    data = generators[args.manifold](num_samples, noise, args.density)
    
    out_path = os.path.join(args.out_dir, "dataset.pt")
    torch.save(data, out_path)
    
    print(f"Manifold tensor saved to {out_path}.")
    print(f"Ambient dimensions: p={data.shape[1]}")
    print(f"Intrinsic dimension strictly required: d={1 if '1d' in args.manifold else 2 if '2d' in args.manifold else 3}")

if __name__ == "__main__":
    main()