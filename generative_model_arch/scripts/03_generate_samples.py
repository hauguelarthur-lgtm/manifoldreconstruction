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
    d = int(config['manifold']['intrinsic_dim'])
    num_samples = int(config['manifold']['num_samples'])

    # Load Full Geometric Artifact Suite
    data_ambient = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
    chart_ambient_indices = torch.load(os.path.join(args.data_dir, "chart_ambient_indices.pt"), map_location='cpu')
    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location=device)
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    z_clusters_intrinsic = torch.load(os.path.join(args.data_dir, "z_clusters_intrinsic.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    
    m = len(whitney_atlas)
    if len(z_clusters_intrinsic) != m:
        raise RuntimeError(f"Fatal Desync: whitney_atlas contains {m} charts, but z_clusters_intrinsic "
                           f"contains {len(z_clusters_intrinsic)} slices. Re-execute scripts/02_solve_velocities.py.")

    time_steps = len(precomputed_etas)
    p_trunc = 256 
    for eta_local in precomputed_etas[0]:
        if eta_local.shape[0] > 0:
            p_trunc = eta_local.shape[0]
            break

    chart_probs = membership_mask.sum(dim=0).float() / membership_mask.sum()
    chart_assignments = torch.multinomial(chart_probs, num_samples, replacement=True)

    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_trunc).to(device)
    wavelet_map_path = os.path.join(args.data_dir, "wavelet_map.pt")
    model.load_state_dict(torch.load(wavelet_map_path, map_location=device))

    Z_0_list = []
    for i in range(m):
        Z_train_i = z_clusters_intrinsic[i]
        U_train_i = chart_intrinsic_coords[i].to(device)
        n_gen_i = int((chart_assignments == i).sum().item())
        
        if Z_train_i.shape[0] > 0 and U_train_i.shape[0] > 0:
            # 1. Total Ambient Empirical Energy (||X_i - mu_i||_F^2)
            idx_i = chart_ambient_indices[i].long().to(device)
            X_i = data_ambient[idx_i]
            mu_i = X_i.mean(dim=0, keepdim=True)
            var_ambient = torch.sum((X_i - mu_i)**2)
            
            # 2. Projected 1st-Order Intrinsic Energy (||U_i||_F^2)
            var_intrinsic = torch.sum(U_train_i**2)
            
            # 3. Captured 2nd-Order Weingarten Energy (||U_quad W_i||_F^2)
            U_quad_i = formulate_quadratic_features(U_train_i)
            W_i = whitney_atlas[i]['W'].to(device)
            var_quad = torch.sum(torch.matmul(U_quad_i, W_i)**2)
            
            # 4. Rigorous Pythagorean Energy Conservation Scalar (\gamma_i)
            if var_intrinsic > 1e-4:
                energy_ratio = (var_ambient - var_quad) / var_intrinsic
                gamma_i = float(torch.sqrt(torch.clamp(energy_ratio, min=1.0)).item())
            else:
                gamma_i = 1.0

            std_U_i = (Z_train_i.std(dim=0, keepdim=True).to(device) * gamma_i) if Z_train_i.shape[0] > 1 else torch.ones((1, d), device=device)
            
            z_min = Z_train_i.min(dim=0).values.to(device) * gamma_i
            z_max = Z_train_i.max(dim=0).values.to(device) * gamma_i
            
            Z_0_raw = torch.randn((n_gen_i, d), device=device) * std_U_i
            Z_0_clamped = torch.clamp(Z_0_raw, min=z_min, max=z_max)
            Z_0_list.append(Z_0_clamped)
        else:
            Z_0_list.append(torch.zeros((n_gen_i, d), device=device))

    print(f"Executing Calibrated RKHS Intrinsic Integration in R^{d}...")
    U_gen_list = generate_samples(Z_0_list, model, precomputed_etas, time_steps, device, args.ode_mode)

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