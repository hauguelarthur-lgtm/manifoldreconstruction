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

    data_ambient = torch.load(os.path.join(args.data_dir, "data.pt"), map_location=device)
    chart_ambient_indices = torch.load(os.path.join(args.data_dir, "chart_ambient_indices.pt"), map_location='cpu')
    whitney_atlas = torch.load(os.path.join(args.data_dir, "whitney_atlas.pt"), map_location='cpu')
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location=device)
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    z_clusters_intrinsic = torch.load(os.path.join(args.data_dir, "z_clusters_intrinsic.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"), map_location=device)
    
    m = len(whitney_atlas)
    time_steps = len(precomputed_etas)

    # --- LOAD CHART-DECOUPLED FEATURE MAPS ---
    decoupled_maps_path = os.path.join(args.data_dir, "wavelet_maps_decoupled.pt")
    global_map_path = os.path.join(args.data_dir, "wavelet_map.pt")
    chart_models = []

    if os.path.exists(decoupled_maps_path):
        print("[DEBUG] Loading chart-decoupled Besov feature maps -> wavelet_maps_decoupled.pt")
        state_dicts = torch.load(decoupled_maps_path, map_location=device)
        for i in range(m):
            if state_dicts[i] is not None:
                p_i = state_dicts[i]['omega'].shape[0]
                model_i = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_i).to(device)
                model_i.load_state_dict(state_dicts[i])
                chart_models.append(model_i)
            else:
                chart_models.append(None)
    elif os.path.exists(global_map_path):
        print("[DEBUG] Loading single global Besov feature map -> wavelet_map.pt (Legacy fallback)")
        global_sd = torch.load(global_map_path, map_location=device)
        p_glob = global_sd['omega'].shape[0]
        glob_model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_glob).to(device)
        glob_model.load_state_dict(global_sd)
        chart_models = [glob_model] * m
    else:
        raise FileNotFoundError("Fatal: Neither wavelet_maps_decoupled.pt nor wavelet_map.pt found. Run scripts/02_solve_velocities.py.")

    chart_probs = membership_mask.sum(dim=0).float() / membership_mask.sum()
    chart_assignments = torch.multinomial(chart_probs, num_samples, replacement=True)

    Z_0_list = []
    for i in range(m):
        Z_train_i = z_clusters_intrinsic[i]
        U_train_i = chart_intrinsic_coords[i].to(device)
        n_gen_i = int((chart_assignments == i).sum().item())
        
        if Z_train_i.shape[0] > 0 and U_train_i.shape[0] > 0 and chart_models[i] is not None:
            idx_i = chart_ambient_indices[i].long().to(device)
            X_i = data_ambient[idx_i]
            mu_i = X_i.mean(dim=0, keepdim=True)
            var_ambient = torch.sum((X_i - mu_i)**2)
            var_intrinsic = torch.sum(U_train_i**2)
            
            W_i = whitney_atlas[i]['W'].to(device)
            U_quad_i = formulate_quadratic_features(U_train_i)
            var_quad = torch.sum(torch.matmul(U_quad_i, W_i)**2)
            
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
    U_gen_list = generate_samples(Z_0_list, chart_models, precomputed_etas, time_steps, device, args.ode_mode)

    print("Executing 2nd-Order Weingarten Affine Lift with Arc-to-Chord Dilation...")
    X_lift_list = []
    for i in range(m):
        if U_gen_list[i].size(0) == 0: continue
        
        U_train_i = chart_intrinsic_coords[i].to(device)
        idx_i = chart_ambient_indices[i].long().to(device)
        X_i = data_ambient[idx_i]
        
        if U_train_i.shape[0] > 1 and X_i.shape[0] > 1:
            mu_i = X_i.mean(dim=0, keepdim=True)
            var_ambient = torch.sum((X_i - mu_i)**2)
            var_intrinsic = torch.sum(U_train_i**2)
            
            W_i = whitney_atlas[i]['W'].to(device)
            U_quad_i = formulate_quadratic_features(U_train_i)
            var_quad = torch.sum(torch.matmul(U_quad_i, W_i)**2)
            
            if var_intrinsic > 1e-4:
                energy_ratio = (var_ambient - var_quad) / var_intrinsic
                gamma_i = float(torch.sqrt(torch.clamp(energy_ratio, min=1.0)).item())
            else:
                gamma_i = 1.0
        else:
            gamma_i = 1.0
            W_i = whitney_atlas[i]['W'].to(device)

        U_gen_dilated = U_gen_list[i] * gamma_i
        U_gen_quad = formulate_quadratic_features(U_gen_dilated)
        
        X_lift = (torch.matmul(U_gen_dilated, whitney_atlas[i]['Q'].to(device).T) + 
                  torch.matmul(U_gen_quad, W_i) + 
                  whitney_atlas[i]['mu'].to(device))
        X_lift_list.append(X_lift)

    output_path = os.path.join(args.data_dir, "generated_samples.pt")
    torch.save(torch.cat(X_lift_list, dim=0).cpu(), output_path)
    print(f"Phase 4 Complete -> {output_path}")

if __name__ == "__main__": main()