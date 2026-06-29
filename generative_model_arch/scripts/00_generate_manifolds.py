import torch
import math
import os
import sys
import argparse



script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

# =====================================================================
# 1D Manifolds (d=1)
# =====================================================================

def generate_simple_1d_helix(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Simple 1D Manifold (d=1, p=3): Helix.
    Teste la torsion constante et la courbure extrinsèque constante.
    """
    t = torch.rand(n_samples) * 10 * math.pi
    x = torch.cos(t)
    y = torch.sin(t)
    z = 0.1 * t
    
    data = torch.stack([x, y, z], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_1d_trefoil(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 1D Manifold (d=1, p=3): Trefoil Knot.
    Possède une courbure extrinsèque très élevée et une auto-intersection apparente.
    """
    t = torch.rand(n_samples) * 2 * math.pi
    x = torch.sin(t) + 2 * torch.sin(2 * t)
    y = torch.cos(t) - 2 * torch.cos(2 * t)
    z = -torch.sin(3 * t)
    
    data = torch.stack([x, y, z], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_1d_lissajous(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 1D Manifold (d=1, p=3): 3D Lissajous Curve.
    Fréquences asymétriques créant des plis spatiaux denses, testant la limite 
    de séparation geodésique vs euclidienne.
    """
    t = torch.rand(n_samples) * 2 * math.pi
    x = torch.sin(3 * t)
    y = torch.sin(4 * t)
    z = torch.cos(5 * t)
    
    data = torch.stack([x, y, z], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

# =====================================================================
# 2D Manifolds (d=2)
# =====================================================================

def generate_simple_2d_sphere(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Simple 2D Manifold (d=2, p=3): 2-Sphere S^2.
    Topologie fermée, courbure gaussienne positive constante.
    """
    data = torch.randn(n_samples, 3)
    data = data / torch.norm(data, dim=1, keepdim=True)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_swiss_roll(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 2D Manifold (d=2, p=3): Swiss Roll.
    Variété ouverte avec distances euclidiennes qui court-circuitent la métrique géodésique.
    """
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n_samples))
    y = 21 * torch.rand(n_samples)
    x = t * torch.cos(t)
    z = t * torch.sin(t)
    
    data = torch.stack([x, y, z], dim=1)
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_hyperbolic_saddle(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 2D Manifold (d=2, p=3): Hyperbolic Paraboloid (Saddle).
    Courbure gaussienne strictement négative. Force le tenseur de Weingarten 
    à modéliser des valeurs propres de signes opposés.
    """
    x = (torch.rand(n_samples) * 2 - 1) * 2
    y = (torch.rand(n_samples) * 2 - 1) * 2
    z = x**2 - y**2
    
    data = torch.stack([x, y, z], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_mobius_strip(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 2D Manifold (d=2, p=3): Möbius Strip.
    Variété non orientable. L'absence de champ de vecteurs normaux globalement continu 
    teste la robustesse de la base PCA locale (U_i) lors des transitions entre cartes.
    """
    u = torch.rand(n_samples) * 2 * math.pi
    v = (torch.rand(n_samples) * 2 - 1) * 0.5  # Width of the strip
    
    x = (1 + v * torch.cos(u / 2)) * torch.cos(u)
    y = (1 + v * torch.cos(u / 2)) * torch.sin(u)
    z = v * torch.sin(u / 2)
    
    data = torch.stack([x, y, z], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_2d_klein_bottle(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex 2D Manifold (d=2, p=4): Flat Klein Bottle Embedding in 4D.
    Surface non orientable sans bords. Plongée dans R^4 pour éviter l'auto-intersection 
    physique qui ruinerait la condition de séparation spatiale de Whitney.
    """
    u = torch.rand(n_samples) * 2 * math.pi
    v = torch.rand(n_samples) * 2 * math.pi
    r1, r2 = 2.0, 1.0
    
    x1 = (r1 + r2 * torch.cos(v)) * torch.cos(u)
    x2 = (r1 + r2 * torch.cos(v)) * torch.sin(u)
    x3 = r2 * torch.sin(v) * torch.cos(u / 2)
    x4 = r2 * torch.sin(v) * torch.sin(u / 2)
    
    data = torch.stack([x1, x2, x3, x4], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

# =====================================================================
# High-Dimensional Manifolds (d > 2)
# =====================================================================

def generate_simple_d3_hypersphere(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Simple D>2 Manifold (d=3, p=4): 3-Sphere S^3.
    Extension dimensionnelle de la sphère.
    """
    data = torch.randn(n_samples, 4)
    data = data / torch.norm(data, dim=1, keepdim=True)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_d3_torus_product(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex D>2 Manifold (d=3, p=5): Product Manifold (T^2 x S^1).
    Topologie de tenseur produit. Un Tore 2D croisé avec un cercle 1D.
    """
    u = torch.rand(n_samples) * 2 * math.pi
    v = torch.rand(n_samples) * 2 * math.pi
    w = torch.rand(n_samples) * 2 * math.pi
    R, r, r2 = 2.0, 0.5, 1.0
    
    x1 = (R + r * torch.cos(v)) * torch.cos(u)
    x2 = (R + r * torch.cos(v)) * torch.sin(u)
    x3 = r * torch.sin(v)
    x4 = r2 * torch.cos(w)
    x5 = r2 * torch.sin(w)
    
    data = torch.stack([x1, x2, x3, x4, x5], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
    return data

def generate_complex_d3_flat_torus(n_samples: int, noise: float = 0.0) -> torch.Tensor:
    """
    Complex D>2 Manifold (d=3, p=6): Flat 3-Torus (S^1 x S^1 x S^1).
    Plongement isométrique dans R^6. Topologie compacte mais courbure intrinsèque nulle.
    Teste si le tenseur local Q parvient à isoler le sous-espace affine R^3.
    """
    u1 = torch.rand(n_samples) * 2 * math.pi
    u2 = torch.rand(n_samples) * 2 * math.pi
    u3 = torch.rand(n_samples) * 2 * math.pi
    
    x1, x2 = torch.cos(u1), torch.sin(u1)
    x3, x4 = torch.cos(u2), torch.sin(u2)
    x5, x6 = torch.cos(u3), torch.sin(u3)
    
    data = torch.stack([x1, x2, x3, x4, x5, x6], dim=1)
    if noise > 0:
        data += torch.randn_like(data) * noise
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
    parser.add_argument("--n_samples", type=int, default=3000, help="Number of empirical points N.")
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of ambient Gaussian noise.")
    parser.add_argument("--out_dir", type=str, default="../generative_model_arch/data/raw", help="Output directory.")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Generating {args.manifold} (N={args.n_samples}, noise={args.noise})...")
    data = generators[args.manifold](args.n_samples, args.noise)
    
    out_path = os.path.join(args.out_dir, "dataset.pt")
    torch.save(data, out_path)
    
    print(f"Manifold tensor saved to {out_path}.")
    print(f"Ambient dimensions: p={data.shape[1]}")
    print(f"Intrinsic dimension strictly required: d={1 if '1d' in args.manifold else 2 if '2d' in args.manifold else 3}")

if __name__ == "__main__":
    main()