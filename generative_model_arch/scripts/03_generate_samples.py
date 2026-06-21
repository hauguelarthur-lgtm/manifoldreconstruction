import torch
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.wavelet_map import TruncatedBesovWaveletMap
from src.SDEintegrator import generate_samples
from src.gluing import apply_terminal_ambient_gluing

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    default_config_path = os.path.join(project_root, "configs", "default_config.yaml")

    parser = argparse.ArgumentParser(description="Executes Intrinsic Flow Integration and Isometric Lift.")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--ode_mode", action="store_true", help="Execute deterministic Heun ODE instead of Euler SDE.")
    parser.add_argument("--glue_ambient", action="store_true", help="Apply terminal partition of unity blending in ambient space.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Config and Artifacts
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    d = config['manifold']['intrinsic_dim']

    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    labels = torch.load(os.path.join(args.data_dir, "labels.pt"), map_location=device)
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    
    num_charts = len(whitney_atlas)
    print(f"Loaded Whitney atlas ({num_charts} charts), intrinsic dimension d={d} on {device}.")

    # 2. Continuous Latent Prior Sampling & Categorical Partitioning
    # Calculate empirical chart occupancy probabilities \pi_i = N_i / N
    chart_counts = torch.bincount(labels, minlength=num_charts).float()
    chart_probs = chart_counts / chart_counts.sum()
    
    # Sample multi-nomial assignments for the new generative batch
    chart_assignments = torch.multinomial(chart_probs, args.num_samples, replacement=True)

    Z_0_list = []
    for i in range(num_charts):
        n_gen_i = (chart_assignments == i).sum().item()
        # Sample strictly from standard continuous normal N(0, I_d)
        Z_0_i = torch.randn((n_gen_i, d), device=device)
        Z_0_list.append(Z_0_i)

    # 3. Instantiate and Calibrate Intrinsic Wavelet Map
    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=args.p_trunc).to(device)
    U_all = torch.cat(chart_intrinsic_coords, dim=0)
    model.calibrate(U_all)

    mode_str = "ODE (Deterministic)" if args.ode_mode else "SDE (Stochastic)"
    print(f"Executing Chart-Decoupled Intrinsic {mode_str} Integration across {args.time_steps} steps...")

    # 4. Perform Intrinsic Integration strictly inside R^d
    U_gen_list = generate_samples(
        Z_0_list=Z_0_list,
        model=model,
        precomputed_etas=precomputed_etas,
        num_time_steps=args.time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    # 5. Exact Isometric Affine Lift to Ambient Space R^p
    print(f"Executing Exact Isometric Affine Lift from R^{d} -> ambient space...")
    X_gen_list = []
    
    for i in range(num_charts):
        U_gen_i = U_gen_list[i]
        if U_gen_i.size(0) == 0:
            continue
            
        mu_i = whitney_atlas[i]['mu'].to(device)
        Q_i = whitney_atlas[i]['Q'].to(device)
        
        # Lift formula: X_i = U_i @ Q_i.T + mu_i
        # Maps (N_gen_i x d) @ (d x p) -> (N_gen_i x p)
        X_gen_i = torch.matmul(U_gen_i, Q_i.T) + mu_i
        X_gen_list.append(X_gen_i)

    X_gen_ambient = torch.cat(X_gen_list, dim=0)

    # 6. Optional Terminal Ambient Gluing
    if args.glue_ambient:
        print("Applying terminal partition of unity blending across ambient chart boundaries...")
        cluster_centers = torch.load(os.path.join(args.data_dir, "cluster_centers.pt"), map_location=device)
        cluster_precisions = torch.load(os.path.join(args.data_dir, "cluster_precisions.pt"), map_location=device)
        X_gen_ambient = apply_terminal_ambient_gluing(X_gen_ambient, whitney_atlas, cluster_centers, cluster_precisions)

    # 7. Serialize Final Generative Tensors
    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(X_gen_ambient.cpu(), output_path)
    print(f"\nPhase 4 Complete. Successfully generated {args.num_samples} ambient samples -> {output_path}")

if __name__ == "__main__":
    main()