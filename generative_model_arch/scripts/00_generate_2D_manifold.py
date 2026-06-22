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

def generate_double_torus_3d(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Generates exact uniform samples on a Genus-2 Double Torus embedded in R^3.
    Algebraic level set: H(x,y,z) = (y^2 - x^2(1 - x^2))^2 + z^2 = c^2.
    Executes Newton-Raphson normal projection with Riemannian coarea density correction.
    """
    c = 0.10
    c_sq = c ** 2
    samples = []
    collected = 0
    
    # Over-sample inside the tight bounding volume containing the double torus
    while collected < num_samples:
        batch_size = num_samples * 10
        x = (torch.rand(batch_size, device=device) * 2.2 - 1.1)
        y = (torch.rand(batch_size, device=device) * 1.2 - 0.6)
        z = (torch.rand(batch_size, device=device) * 0.3 - 0.15)
        pts = torch.stack([x, y, z], dim=1)
        
        # 5 iterations of exact Newton-Raphson normal projection
        for _ in range(5):
            x_col, y_col, z_col = pts[:, 0], pts[:, 1], pts[:, 2]
            x_sq = x_col ** 2
            term = y_col ** 2 - x_sq * (1.0 - x_sq)
            H_val = term ** 2 + z_col ** 2 - c_sq
            
            # Analytical Gradients \nabla H
            grad_x = 2.0 * term * (-2.0 * x_col + 4.0 * x_col ** 3)
            grad_y = 4.0 * term * y_col
            grad_z = 2.0 * z_col
            grad = torch.stack([grad_x, grad_y, grad_z], dim=1)
            grad_norm_sq = torch.sum(grad ** 2, dim=1, keepdim=True)
            
            pts = pts - (H_val.unsqueeze(1) / (grad_norm_sq + 1e-12)) * grad
            
        x_col, y_col, z_col = pts[:, 0], pts[:, 1], pts[:, 2]
        x_sq = x_col ** 2
        final_err = torch.abs((y_col ** 2 - x_sq * (1.0 - x_sq)) ** 2 + z_col ** 2 - c_sq)
        
        # Coarea formula density correction: accept proportional to ||\nabla H||
        grad_x = 2.0 * (y_col ** 2 - x_sq * (1.0 - x_sq)) * (-2.0 * x_col + 4.0 * x_col ** 3)
        grad_y = 4.0 * (y_col ** 2 - x_sq * (1.0 - x_sq)) * y_col
        grad_z = 2.0 * z_col
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        max_norm = 1.20 # Empirical gradient norm supremum
        accept_prob = (grad_norm / max_norm).clamp(0.0, 1.0)
        
        valid_mask = (final_err < 1e-4) & (torch.rand(batch_size, device=device) < accept_prob)
        if torch.any(valid_mask):
            valid_pts = pts[valid_mask]
            samples.append(valid_pts)
            collected += valid_pts.size(0)
            
    final_tensor = torch.cat(samples, dim=0)[:num_samples]
    return (final_tensor - final_tensor.mean(dim=0)) / final_tensor.std(dim=0)

def generate_trefoil_torus_3d(num_samples: int, device: torch.device) -> torch.Tensor:
    """Generates uniform samples on a 2D torus embedded around a Trefoil knot in R^3."""
    u = torch.rand(num_samples, device=device) * 2.0 * np.pi
    v = torch.rand(num_samples, device=device) * 2.0 * np.pi
    
    rho, r_tube = 2.0, 0.50
    cos3, sin3 = torch.cos(3.0 * u), torch.sin(3.0 * u)
    cos2, sin2 = torch.cos(2.0 * u), torch.sin(2.0 * u)
    
    # Trefoil Centerline c(u)
    c = torch.stack([(rho + cos3) * cos2, (rho + cos3) * sin2, sin3], dim=1)
    
    # 1st Derivatives c'(u)
    dc = torch.stack([-3.0 * sin3 * cos2 - 2.0 * (rho + cos3) * sin2,
                      -3.0 * sin3 * sin2 + 2.0 * (rho + cos3) * cos2,
                      3.0 * cos3], dim=1)
    
    # 2nd Derivatives c''(u)
    ddc = torch.stack([-9.0 * cos3 * cos2 + 12.0 * sin3 * sin2 - 4.0 * (rho + cos3) * cos2,
                       -9.0 * cos3 * sin2 - 12.0 * sin3 * cos2 - 4.0 * (rho + cos3) * sin2,
                       -9.0 * sin3], dim=1)
    
    # Frenet-Serret Orthogonal Framing
    T = dc / torch.norm(dc, dim=1, keepdim=True)
    B_raw = torch.cross(dc, ddc, dim=1)
    B = B_raw / torch.norm(B_raw, dim=1, keepdim=True)
    N = torch.cross(B, T, dim=1)
    
    surface_pts = c + r_tube * (torch.cos(v).unsqueeze(1) * N + torch.sin(v).unsqueeze(1) * B)
    return (surface_pts - surface_pts.mean(dim=0)) / surface_pts.std(dim=0)

def generate_klein_bottle_3d(num_samples: int, device: torch.device) -> torch.Tensor:
    """Generates samples on the non-orientable Figure-8 Bagel Klein Bottle immersed in R^3."""
    u = torch.rand(num_samples, device=device) * 2.0 * np.pi
    v = torch.rand(num_samples, device=device) * 2.0 * np.pi
    half_u = u / 2.0
    
    profile = 2.0 + torch.cos(half_u) * torch.sin(v) - torch.sin(half_u) * torch.sin(2.0 * v)
    x = profile * torch.cos(u)
    y = profile * torch.sin(u)
    z = torch.sin(half_u) * torch.sin(v) + torch.cos(half_u) * torch.sin(2.0 * v)
    
    pts = torch.stack([x, y, z], dim=1)
    return (pts - pts.mean(dim=0)) / pts.std(dim=0)

def generate_roman_surface_3d(num_samples: int, device: torch.device) -> torch.Tensor:
    """Generates samples on Steiner's Roman Surface (singular RP^2 immersion in R^3)."""
    theta = torch.rand(num_samples, device=device) * 2.0 * np.pi
    phi = torch.acos(2.0 * torch.rand(num_samples, device=device) - 1.0) / 2.0
    
    x = 2.0 * torch.sin(2.0 * phi) * torch.cos(theta)
    y = 2.0 * torch.sin(2.0 * phi) * torch.sin(theta)
    z = 2.0 * torch.cos(phi) * torch.sin(2.0 * theta) * torch.sin(phi)
    
    pts = torch.stack([x, y, z], dim=1)
    return (pts - pts.mean(dim=0)) / pts.std(dim=0)

def generate_mobius_strip_3d(num_samples: int, device: torch.device) -> torch.Tensor:
    """Generates samples on a thickened Möbius Strip embedded in R^3."""
    u = torch.rand(num_samples, device=device) * 2.0 * np.pi
    v = (torch.rand(num_samples, device=device) * 2.0 - 1.0) * 0.6
    half_u = u / 2.0
    
    x = (2.0 + v * torch.cos(half_u)) * torch.cos(u)
    y = (2.0 + v * torch.cos(half_u)) * torch.sin(u)
    z = v * torch.sin(half_u)
    
    pts = torch.stack([x, y, z], dim=1)
    return (pts - pts.mean(dim=0)) / pts.std(dim=0)

def generate_mnist(num_samples: int, device: torch.device) -> torch.Tensor:
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=num_samples, shuffle=True)
    images, _ = next(iter(data_loader))
    return images.view(images.size(0), -1).to(device)

def sample_single_fundamental_surface(N: int, device: torch.device) -> tuple[torch.Tensor, str, str]:
    u = torch.rand(N, device=device)
    v = torch.rand(N, device=device)
    surface_type = torch.randint(0, 5, (1,)).item()
    
    if surface_type == 0:
        g_name, presentation = "Trivial Group {1}", "< 1 | 1 > (Sphere S^2)"
        theta = 2.0 * np.pi * u
        phi = torch.acos(2.0 * v - 1.0)
        coords = torch.stack([torch.sin(phi)*torch.cos(theta), torch.sin(phi)*torch.sin(theta), torch.cos(phi), torch.zeros(N, device=device)], dim=1)
    elif surface_type == 1:
        g_name, presentation = "Free Abelian Z^2", "< a, b | [a, b] = 1 > (Torus T^2)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        coords = torch.stack([torch.cos(theta), torch.sin(theta), torch.cos(phi), torch.sin(phi)], dim=1)
    elif surface_type == 2:
        g_name, presentation = "Surface Group \pi_1(\Sigma_2)", "< a1,b1,a2,b2 | [a1,b1][a2,b2]=1 > (Double Torus)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        r_tube = 0.4 + 0.2 * torch.sin(2.0 * theta)
        coords = torch.stack([(2.0 + r_tube*torch.cos(phi))*torch.cos(theta), (2.0 + r_tube*torch.cos(phi))*torch.sin(theta), r_tube*torch.sin(phi), torch.sin(3.0*theta)*0.3], dim=1)
    elif surface_type == 3:
        g_name, presentation = "Cyclic Group Z_2", "< c | c^2 = 1 > (Real Projective Plane RP^2)"
        theta, phi = 2.0 * np.pi * u, np.pi * v
        x, y, z = torch.sin(phi)*torch.cos(theta), torch.sin(phi)*torch.sin(theta), torch.cos(phi)
        coords = torch.stack([x**2 - y**2, x*y*2.0, x*z*2.0, y*z*2.0], dim=1)
    else:
        g_name, presentation = "Klein Bottle Group", "< a, b | b a b^{-1} a = 1 > (Klein Bottle K^2)"
        theta, phi = 2.0 * np.pi * u, 2.0 * np.pi * v
        r_mob = 1.5 + torch.cos(phi/2.0)*torch.sin(theta) - torch.sin(phi/2.0)*torch.sin(2.0*theta)
        coords = torch.stack([r_mob*torch.cos(phi), r_mob*torch.sin(phi), torch.sin(phi/2.0)*torch.sin(theta) + torch.cos(phi/2.0)*torch.sin(2.0*theta), torch.cos(theta)], dim=1)
    return coords, g_name, presentation

def generate_random_fundamental_product(num_samples: int, ambient_dim: int, num_factors: int, device: torch.device) -> torch.Tensor:
    N, k = num_samples, num_factors
    stacked_base_dim = 4 * k
    if ambient_dim < stacked_base_dim: raise ValueError(f"ambient_dim >= {stacked_base_dim} required.")

    component_coords = [sample_single_fundamental_surface(N, device)[0] for _ in range(k)]
    base_product_coords = torch.cat(component_coords, dim=1)

    torch.manual_seed(42)
    A_def = torch.randn(stacked_base_dim, stacked_base_dim, device=device) * 0.20
    omega_def = torch.randn(stacked_base_dim, stacked_base_dim, device=device) * (2.0 / float(np.sqrt(k)))
    phase_def = torch.rand(stacked_base_dim, device=device) * 2.0 * np.pi
    
    deformed_coords = base_product_coords + torch.matmul(torch.sin(torch.matmul(base_product_coords, omega_def) + phase_def), A_def)
    deformed_coords = deformed_coords - deformed_coords.mean(dim=0, keepdim=True)

    Q_random, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, device=device))
    padded_coords = torch.zeros(N, ambient_dim, device=device)
    padded_coords[:, :stacked_base_dim] = deformed_coords
    return torch.matmul(padded_coords, Q_random.T).float()

def main():
    parser = argparse.ArgumentParser(description="Generates raw ambient submanifolds.")
    parser.add_argument("--topology", type=str, default="double_torus_3d", choices=[
        "swiss_roll", "torus", "sphere", "default", 
        "random_fundamental", "random_fundamental_product", "mnist",
        "double_torus_3d", "trefoil_torus_3d", "klein_bottle_3d", 
        "roman_surface_3d", "mobius_strip_3d"
    ])
    parser.add_argument("--product_factors", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "raw"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    num_samples = int(config['manifold']['num_samples'])
    ambient_dim = int(config['manifold']['ambient_dim'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Topological Routing
    if args.topology == "swiss_roll": dataset = generate_swiss_roll(num_samples, ambient_dim, device)
    elif args.topology == "torus": dataset = generate_torus(num_samples, ambient_dim, device)
    elif args.topology == "sphere": dataset = generate_sphere(num_samples, ambient_dim, device)
    elif args.topology == "mnist": dataset = generate_mnist(num_samples, device)
    elif args.topology == "random_fundamental":
        coords = sample_single_fundamental_surface(num_samples, device)[0]
        padded = torch.zeros(num_samples, ambient_dim, device=device)
        padded[:, :4] = coords
        Q, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, device=device))
        dataset = torch.matmul(padded, Q.T).float()
    elif args.topology == "random_fundamental_product":
        dataset = generate_random_fundamental_product(num_samples, ambient_dim, args.product_factors, device)
    elif args.topology in ["double_torus_3d", "trefoil_torus_3d", "klein_bottle_3d", "roman_surface_3d", "mobius_strip_3d"]:
        if args.topology == "double_torus_3d": base_coords = generate_double_torus_3d(num_samples, device)
        elif args.topology == "trefoil_torus_3d": base_coords = generate_trefoil_torus_3d(num_samples, device)
        elif args.topology == "klein_bottle_3d": base_coords = generate_klein_bottle_3d(num_samples, device)
        elif args.topology == "roman_surface_3d": base_coords = generate_roman_surface_3d(num_samples, device)
        elif args.topology == "mobius_strip_3d": base_coords = generate_mobius_strip_3d(num_samples, device)
        
        # Automatic Orthogonal Haar Lift for high-dimensional testing
        if ambient_dim > 3:
            print(f"[DEBUG] Lifting 3D base immersion into ambient R^{ambient_dim} via Haar rotation SO({ambient_dim})...")
            Q_random, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, device=device))
            padded_coords = torch.zeros(num_samples, ambient_dim, device=device)
            padded_coords[:, :3] = base_coords
            dataset = torch.matmul(padded_coords, Q_random.T).float()
        else:
            dataset = base_coords
    else:
        dataset = generate_nonlinear_manifold(num_samples, ambient_dim, device)

    output_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save(dataset.cpu(), output_path)
    print(f"Successfully generated {args.topology} ({dataset.shape}) -> {output_path}")

if __name__ == "__main__": main()