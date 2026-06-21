import torch
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.wavelet_map import TruncatedBesovWaveletMap
from src.SDEintegrator import generate_samples
from src.gluing import compute_subordinated_partition_of_unity

def formulate_quadratic_features(U: torch.Tensor) -> torch.Tensor:
    """
    Generates exact upper-triangular quadratic outer products of intrinsic coordinates.
    Matches the exact feature order solved during Phase 1 Weingarten regression.
    """
    N, d = U.shape
    quad_dim = d * (d + 1) // 2
    U_quad = torch.zeros(N, quad_dim, device=U.device)
    col = 0
    for dim1 in range(d):
        for dim2 in range(dim1, d):
            U_quad[:, col] = U[:, dim1] * U[:, dim2]
            col += 1
    return U_quad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--ode_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    d = config['manifold']['intrinsic_dim']

    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    smooth_sigmas = torch.load(os.path.join(args.data_dir, "smooth_sigmas.pt"), map_location=device)
    
    m = len(whitney_atlas)
    chart_radii = torch.sqrt(smooth_sigmas)
    
    # Extract intrinsic chart centroids
    centroids = torch.stack([chart_intrinsic_coords[i].mean(dim=0) for i in range(m)]).to(device)

    # 1. Sample Master Continuous Latent Prior in R^d
    torch.manual_seed(42)
    Z_0 = torch.randn(args.num_samples, d, device=device)

    # 2. Instantiate and Calibrate Besov Wavelet Map
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=args.p_trunc).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))

    print(f"Executing Compact-Weighted Intrinsic SDE Integration in R^{d}...")
    U_gen = generate_samples(
        Z_0=Z_0,
        model=model,
        precomputed_etas=precomputed_etas,
        centroids=centroids,
        chart_radii=chart_radii,
        num_time_steps=args.time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    # 3. Exact 2nd-Order Weingarten Lift to Ambient Space R^16
    print("Executing 2nd-Order Weingarten Affine Lift (Quadratic Normal Correction)...")
    X_gen_ambient = torch.zeros(args.num_samples, 16, device=device)
    
    # Evaluate subordinated compact weights to blend the lifted ambient patches
    terminal_weights = compute_subordinated_partition_of_unity(U_gen, centroids, chart_radii)

    U_quad_gen = formulate_quadratic_features(U_gen)

    for i in range(m):
        mu_i = whitney_atlas[i]['mu'].to(device)
        Q_i = whitney_atlas[i]['Q'].to(device)
        W_i = whitney_atlas[i]['W'].to(device)
        
        # EXACT CORRECTION: 2nd-Order Lift Formula (arXiv:2506.19587)
        # X_i = U @ Q.T + U_quad @ W + mu
        X_lift_i = torch.matmul(U_gen, Q_i.T) + torch.matmul(U_quad_gen, W_i) + mu_i
        
        rho_i = terminal_weights[:, i].unsqueeze(1)
        X_gen_ambient += rho_i * X_lift_i

    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(X_gen_ambient.cpu(), output_path)
    print(f"Phase 4 Complete. Serialized mathematically exact 2nd-order samples -> {output_path}")

if __name__ == "__main__":
    main()