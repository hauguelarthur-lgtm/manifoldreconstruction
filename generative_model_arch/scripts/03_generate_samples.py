import math
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

def formulate_quadratic_features(U: torch.Tensor) -> torch.Tensor:
    """Generates exact upper-triangular quadratic outer products of intrinsic coordinates."""
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
    parser.add_argument("--ode_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    d = config['manifold']['intrinsic_dim']
    num_samples=config['manifold']['num_samples']
    p_trunc=config['features']['p_trunc']
    time_steps=config['integration']['time_steps']

    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location=device)
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    smooth_sigmas = torch.load(os.path.join(args.data_dir, "smooth_sigmas.pt"), map_location='cpu')
    
    m = len(whitney_atlas)
    # Covering radius r = 1.5 * delta extracted from Phase 1 sigma parameterization
    covering_radius = math.sqrt(smooth_sigmas[0].item()) * 2.0  

    # 1. Categorical Chart Partitioning of the Generative Batch
    chart_counts = membership_mask.sum(dim=0).float()
    chart_probs = chart_counts / chart_counts.sum()
    
    torch.manual_seed(42)
    chart_assignments = torch.multinomial(chart_probs, num_samples, replacement=True)

    z_clusters = torch.load(os.path.join(args.data_dir, "z_clusters_intrinsic.pt"), map_location=device)
    
    Z_0_list = []
    for i in range(m):
        z_i = z_clusters[i].to(device)
        print(f"Chart {i} Prior Mean: {z_i.mean(dim=0)}")
        # Compute empirical statistics of the training prior for this chart
        mean_i = z_i.mean(dim=0)
        std_i = z_i.std(dim=0)
        
        n_gen_i = (chart_assignments == i).sum().item()
        # Sample inference priors from the same support as training priors
        Z_0_i = torch.normal(mean=mean_i, std=std_i.expand(n_gen_i, d))
        Z_0_list.append(Z_0_i)


    '''for i in range(m):
        z_i = z_clusters[i].to(device)
        z_min = z_i.min(dim=0).values
        z_max = z_i.max(dim=0).values
        mean_i = z_i.mean(dim=0)
        std_i = z_i.std(dim=0)
    
        # 2. Sample and reject out-of-bounds
        n_gen_i = (chart_assignments == i).sum().item()
        Z_0_i = torch.normal(mean=mean_i, std=std_i.expand(n_gen_i, d))
    
        # Truncate particles that drift into the "dead zone" where vector field is undefined
        mask = (Z_0_i >= z_min) & (Z_0_i <= z_max)
        # Force into bounds
        Z_0_i = torch.clamp(Z_0_i, min=z_min, max=z_max)
        Z_0_list.append(Z_0_i)'''


    # 2. Instantiate and Calibrate Besov Wavelet Map
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_trunc).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))

    print(f"Executing Chart-Decoupled Intrinsic SDE Integration in R^{d}...")
    U_gen_list = generate_samples(
        Z_0_list=Z_0_list,
        model=model,
        precomputed_etas=precomputed_etas,
        num_time_steps=time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    # 3. Exact 2nd-Order Weingarten Affine Lift to Ambient Space R^16
    print("Executing 2nd-Order Weingarten Affine Lift (Quadratic Normal Correction)...")
    X_lift_list = []

    for i in range(m):
        U_gen_i = U_gen_list[i]
        if U_gen_i.size(0) == 0:
            continue
            
        mu_i = whitney_atlas[i]['mu'].to(device)
        Q_i = whitney_atlas[i]['Q'].to(device)
        W_i = whitney_atlas[i]['W'].to(device)
        print(W_i.norm())
        
        U_quad_i = formulate_quadratic_features(U_gen_i)
        
        # EXACT CORRECTION: 2nd-Order Lift Formula (arXiv:2506.19587)
        # X_i = U @ Q.T + U_quad @ W + mu
        X_lift_i = torch.matmul(U_gen_i, Q_i.T) + torch.matmul(U_quad_i, W_i) + mu_i
        X_lift_list.append(X_lift_i)

    X_gen_ambient = torch.cat(X_lift_list, dim=0)


    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(X_gen_ambient.cpu(), output_path)
    print(f"Phase 4 Complete. Serialized mathematically exact 2nd-order ambient samples -> {output_path}")

if __name__ == "__main__":
    main()