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
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    d = config['manifold']['intrinsic_dim']

    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location=device)
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    z_clusters_intrinsic = torch.load(os.path.join(args.data_dir, "z_clusters_intrinsic.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    
    m = len(whitney_atlas)
    chart_probs = membership_mask.sum(dim=0).float() / membership_mask.sum()
    chart_assignments = torch.multinomial(chart_probs, config['manifold']['num_samples'], replacement=True)

    # MATHEMATICAL FIX 5: Calibrated RKHS Support-Confined KDE Prior Sampling
    # Resamples exact Phase 2 prior anchors with smooth Kernel smoothing jitter.
    # Confines test particles strictly inside the trained RKHS transport tube.
    Z_0_list = []
    for i in range(m):
        Z_train_i = z_clusters_intrinsic[i]
        n_gen_i = (chart_assignments == i).sum().item()
        
        if Z_train_i.shape[0] > 0:
            # Resample training anchors with replacement
            idx = torch.randint(0, Z_train_i.shape[0], (n_gen_i,), device=device)
            base_anchors = Z_train_i[idx]
            
            # Add continuous Kernel Density smoothing jitter (sigma_kde = 0.05 * std)
            kde_bandwidth = (Z_train_i.std(dim=0, keepdim=True) + 1e-6) * 0.05
            jitter = torch.randn((n_gen_i, d), device=device) * kde_bandwidth
            Z_0_list.append(base_anchors + jitter)
        else:
            Z_0_list.append(torch.zeros((n_gen_i, d), device=device))

    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=config['features']['p_trunc']).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))

    print(f"Executing Calibrated RKHS Intrinsic Integration in R^{d}...")
    U_gen_list = generate_samples(Z_0_list, model, precomputed_etas, config['integration']['time_steps'], device, args.ode_mode)

    print("Executing 2nd-Order Weingarten Affine Lift...")
    X_lift_list = []
    for i in range(m):
        if U_gen_list[i].size(0) == 0: continue
        X_lift_list.append(torch.matmul(U_gen_list[i], whitney_atlas[i]['Q'].to(device).T) + 
                           torch.matmul(formulate_quadratic_features(U_gen_list[i]), whitney_atlas[i]['W'].to(device)) + 
                           whitney_atlas[i]['mu'].to(device))

    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(torch.cat(X_lift_list, dim=0).cpu(), output_path)
    print(f"Phase 4 Complete -> {output_path}")

if __name__ == "__main__": main()