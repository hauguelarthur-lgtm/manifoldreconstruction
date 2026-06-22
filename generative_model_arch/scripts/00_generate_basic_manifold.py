import torch
import os
import sys
import argparse
import yaml
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

def generate_swiss_roll(num_samples: int, ambient_dim: int, device: torch.device) -> torch.Tensor:
    t = 1.5 * np.pi * (1.0 + 2.0 * torch.rand(num_samples, device=device))
    height = 21.0 * torch.rand(num_samples, device=device)
    coords = torch.zeros(num_samples, ambient_dim, device=device)
    coords[:, 0] = t * torch.cos(t)
    coords[:, 1] = height
    coords[:, 2] = t * torch.sin(t)
    return coords

def generate_torus(num_samples: int, ambient_dim: int, device: torch.device) -> torch.Tensor:
    u = torch.rand(num_samples, device=device) * 2.0 * np.pi
    v = torch.rand(num_samples, device=device) * 2.0 * np.pi
    R, r = 2.0, 0.5
    coords = torch.zeros(num_samples, ambient_dim, device=device)
    coords[:, 0] = (R + r * torch.cos(v)) * torch.cos(u)
    coords[:, 1] = (R + r * torch.cos(v)) * torch.sin(u)
    coords[:, 2] = r * torch.sin(v)
    return coords

def generate_sphere(num_samples: int, ambient_dim: int, device: torch.device) -> torch.Tensor:
    u = torch.rand(num_samples, device=device)
    v = torch.rand(num_samples, device=device)
    theta = u * 2.0 * np.pi
    phi = torch.acos(2.0 * v - 1.0)
    coords = torch.zeros(num_samples, ambient_dim, device=device)
    coords[:, 0] = torch.sin(phi) * torch.cos(theta)
    coords[:, 1] = torch.sin(phi) * torch.sin(theta)
    coords[:, 2] = torch.cos(phi)
    return coords

def generate_nonlinear_manifold(num_samples: int, ambient_dim: int, device: torch.device) -> torch.Tensor:
    d = 3
    if ambient_dim < 3 * d:
        raise ValueError(f"Nonlinear manifold requires ambient_dim >= {3 * d}")
    Z = torch.randn(num_samples, d, device=device)
    X = torch.zeros(num_samples, ambient_dim, device=device)
    X[:, :d] = Z
    for j in range(d):
        X[:, d + j] = torch.sin(2.0 * Z[:, j])
        X[:, 2 * d + j] = torch.cos(3.0 * Z[:, j])
    return X

def sample_single_fundamental_surface(N: int, device: torch.device) -> tuple[torch.Tensor, str, str]:
    """
    Samples a single compact 2D Riemann surface immersion in base R^4.
    UPGRADED: Injects severe Extrinsic/Immersion Complexity to challenge the 
    Fefferman-Whitney local tangent frames and Weingarten normal regression.
    """
    u = torch.rand(N, device=device)
    v = torch.rand(N, device=device)
    
    # Expanded topological pool: 0-4 (Foundational), 5-7 (Advanced Complex)
    surface_type = torch.randint(0, 8, (1,)).item()
    
    if surface_type == 0: # Sphere S^2
        g_name, presentation = "Trivial Group {1}", "< 1 | 1 > (Sphere S^2)"
        theta = 2.0 * np.pi * u
        phi = torch.acos(2.0 * v - 1.0)
        coords = torch.stack([torch.sin(phi)*torch.cos(theta), torch.sin(phi)*torch.sin(theta), torch.cos(phi), torch.zeros(N, device=device)], dim=1)
        
    elif surface_type == 1: # Torus T^2
        g_name, presentation = "Free Abelian Z^2", "< a, b | [a, b] = 1 > (Torus T^2)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        coords = torch.stack([torch.cos(theta), torch.sin(theta), torch.cos(phi), torch.sin(phi)], dim=1)
        
    elif surface_type == 2: # Double Torus \Sigma_2
        g_name, presentation = "Surface Group \pi_1(\Sigma_2)", "< a1,b1,a2,b2 | [a1,b1][a2,b2]=1 > (Double Torus)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        r_tube = 0.4 + 0.2 * torch.sin(2.0 * theta)
        coords = torch.stack([(2.0 + r_tube*torch.cos(phi))*torch.cos(theta), (2.0 + r_tube*torch.cos(phi))*torch.sin(theta), r_tube*torch.sin(phi), torch.sin(3.0*theta)*0.3], dim=1)
        
    elif surface_type == 3: # Real Projective Plane RP^2
        g_name, presentation = "Cyclic Group Z_2", "< c | c^2 = 1 > (Real Projective Plane RP^2)"
        theta, phi = 2.0 * np.pi * u, np.pi * v
        x, y, z = torch.sin(phi)*torch.cos(theta), torch.sin(phi)*torch.sin(theta), torch.cos(phi)
        coords = torch.stack([x**2 - y**2, x*y*2.0, x*z*2.0, y*z*2.0], dim=1)
        
    elif surface_type == 4: # Klein Bottle K^2
        g_name, presentation = "Klein Bottle Group", "< a, b | b a b^{-1} a = 1 > (Klein Bottle K^2)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        r_mob = 1.5 + torch.cos(phi/2.0)*torch.sin(theta) - torch.sin(phi/2.0)*torch.sin(2.0*theta)
        coords = torch.stack([r_mob*torch.cos(phi), r_mob*torch.sin(phi), torch.sin(phi/2.0)*torch.sin(theta) + torch.cos(phi/2.0)*torch.sin(2.0*theta), torch.cos(theta)], dim=1)

    elif surface_type == 5: # Steiner's Roman Surface (Complex RP^2 Immersion)
        g_name, presentation = "Roman Surface (RP^2)", "< c | c^2 = 1 > (Whitney Umbrella Extrinsic Geometry)"
        theta, phi = 2.0 * np.pi * u, np.pi * v
        x = torch.sin(2.0 * phi) * torch.cos(theta)
        y = torch.sin(2.0 * phi) * torch.sin(theta)
        z = torch.cos(phi) * torch.sin(2.0 * theta) * torch.sin(phi)
        w = torch.cos(2.0 * phi) # Orthogonal lift resolves the central 3D singularity
        coords = torch.stack([x, y, z, w], dim=1) * 2.0

    elif surface_type == 6: # Trefoil Knotted Torus (Extrinsically Complex T^2)
        g_name, presentation = "Knotted Torus T^2", "< a, b | [a, b] = 1 > (Trefoil Knot Embedding)"
        t, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        # Base Trefoil Knot
        r_k = 2.0 + torch.cos(3.0 * t)
        x_k = r_k * torch.cos(2.0 * t)
        y_k = r_k * torch.sin(2.0 * t)
        z_k = torch.sin(3.0 * t)
        # Tubular Frenet-Serret normal extrusion into R^4
        tube_r = 0.6
        coords = torch.stack([
            x_k + tube_r * torch.cos(phi), 
            y_k + tube_r * torch.sin(phi), 
            z_k + tube_r * torch.cos(phi) * torch.sin(t), 
            tube_r * torch.sin(phi) * torch.cos(t)
        ], dim=1)

    else: # 4D Lissajous Torus (Fully non-degenerate 4D curvatures)
        g_name, presentation = "4D Lissajous Torus", "< a, b | [a, b] = 1 > (Orthogonally Distributed Curvature)"
        # Rotates the principal curvatures explicitly into mutually orthogonal planes
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        coords = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi)
        ], dim=1) * 2.0
        
    return coords, g_name, presentation


    
def generate_random_fundamental_product(num_samples: int, ambient_dim: int, num_factors: int, device: torch.device) -> torch.Tensor:
    """
    Constructs an exact Cartesian product manifold M_1 x M_2 x ... x M_k 
    by orthogonally stacking sampled fundamental groups in base R^(4k).
    """
    N = num_samples
    k = num_factors
    stacked_base_dim = 4 * k
    
    if ambient_dim < stacked_base_dim:
        raise ValueError(f"Fatal: ambient_dim ({ambient_dim}) must be >= 4 * num_factors ({stacked_base_dim}) "
                         f"to support orthogonal subspace immersion stacking.")

    print(f"\n[DEBUG] --- CONSTRUCTING CARTESIAN PRODUCT OF {k} FUNDAMENTAL GROUPS ---")
    component_coords = []
    group_names = []
    
    for r in range(k):
        coords_r, g_name_r, pres_r = sample_single_fundamental_surface(N, device)
        component_coords.append(coords_r)
        group_names.append(g_name_r)
        print(f"[DEBUG] Factor {r+1} (\pi_1) : {g_name_r} | Presentation: {pres_r}")

    print(f"[DEBUG] Direct Product \pi_1(M) : {' x '.join(group_names)}")
    print(f"[DEBUG] True Intrinsic Rank (d): {2 * k}")

    # 1. Orthogonal Subspace Stacking along columns
    base_product_coords = torch.cat(component_coords, dim=1)  # Shape: (N, 4k)

    # 2. Coupled Multiscale Diffeomorphic Metric Dilation G_{ij}(x)
    print(f"[DEBUG] Applying coupled multiscale Fourier metric deformation across R^{stacked_base_dim}...")
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())
    A_def = torch.randn(stacked_base_dim, stacked_base_dim, device=device) * 0.20
    omega_def = torch.randn(stacked_base_dim, stacked_base_dim, device=device) * (2.0 / float(np.sqrt(k)))
    phase_def = torch.rand(stacked_base_dim, device=device) * 2.0 * np.pi
    
    deformed_coords = base_product_coords + torch.matmul(torch.sin(torch.matmul(base_product_coords, omega_def) + phase_def), A_def)
    deformed_coords = deformed_coords - deformed_coords.mean(dim=0, keepdim=True)

    # 3. Haar Orthogonal Ambient Entanglement SO(p)
    print(f"[DEBUG] Lifting {2*k}D intrinsic product into ambient R^{ambient_dim} via Haar rotation SO({ambient_dim})...")
    Q_random, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, device=device))
    
    padded_coords = torch.zeros(N, ambient_dim, device=device)
    padded_coords[:, :stacked_base_dim] = deformed_coords

    X_ambient = torch.matmul(padded_coords, Q_random.T)
    return X_ambient.float()

def main():
    parser = argparse.ArgumentParser(description="Generates raw ambient submanifolds.")
    parser.add_argument("--topology", type=str, default="default", choices=["swiss_roll", "torus", "sphere", "default", "random_fundamental", "random_fundamental_product"])
    parser.add_argument("--product_factors", type=int, default=2, help="Number of Cartesian product factors k (Intrinsic rank = 2k).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "raw"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    num_samples = int(config['manifold']['num_samples'])
    ambient_dim = int(config['manifold']['ambient_dim'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.topology == "swiss_roll":
        dataset = generate_swiss_roll(num_samples, ambient_dim, device)
    elif args.topology == "torus":
        dataset = generate_torus(num_samples, ambient_dim, device)
    elif args.topology == "sphere":
        dataset = generate_sphere(num_samples, ambient_dim, device)
    elif args.topology == "random_fundamental":
        coords, _, _ = sample_single_fundamental_surface(num_samples, device)
        padded = torch.zeros(num_samples, ambient_dim, device=device)
        padded[:, :4] = coords
        Q, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, device=device))
        dataset = torch.matmul(padded, Q.T).float()
    elif args.topology == "random_fundamental_product":
        dataset = generate_random_fundamental_product(num_samples, ambient_dim, args.product_factors, device)
    else:
        dataset = generate_nonlinear_manifold(num_samples, ambient_dim, device)

    output_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save(dataset.cpu(), output_path)
    print(f"Successfully generated {args.topology} ({dataset.shape}) -> {output_path}")

if __name__ == "__main__": main()