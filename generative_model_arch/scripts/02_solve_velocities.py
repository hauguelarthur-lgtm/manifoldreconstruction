import torch
import os
import sys
import argparse
import ot
import yaml
import numpy as np
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.wavelet_map import TruncatedBesovWaveletMap
from src.local_linear_regression import solve_local_system

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    membership_mask = torch.load(os.path.join(args.data_dir, "membership_mask.pt"), map_location='cpu')
    chart_intrinsic_coords = torch.load(os.path.join(args.data_dir, "chart_intrinsic_coords.pt"), map_location=device)
    
    m = int(membership_mask.shape[1])
    d = int(chart_intrinsic_coords[0].shape[1])
    N = int(membership_mask.shape[0])

    beta_path = os.path.join(args.data_dir, "besov_beta.pt")
    if os.path.exists(beta_path):
        besov_beta = float(torch.load(beta_path, map_location='cpu').item())
    else:
        besov_beta = 1.50

    raw_time_steps = config['integration']['time_steps']
    if str(raw_time_steps).strip().lower() in ['auto', 'none', '0']:
        exponent = besov_beta / (2.0 * besov_beta + float(d))
        t_calc = int(15.0 * math.pow(N, exponent))
        time_steps = max(50, min(t_calc, 2000)) * 2
    else:
        time_steps = int(float(raw_time_steps))

    # =========================================================
    # PILLAR 1: OPTIMAL TRANSPORT LOOP (With Chart Advancement)
    # =========================================================
    print(f"Executing Covariance-Matched Exact Optimal Transport across {m} charts in R^{d}...")
    z_clusters_intrinsic = []
    torch.manual_seed(42)

    for i in range(m):
        # In-place terminal advancement update
        sys.stdout.write(f"\r[Optimal Transport] Solved chart {i + 1}/{m} ({(i + 1) / m * 100:.1f}%)")
        sys.stdout.flush()

        U_i = chart_intrinsic_coords[i]
        N_i = U_i.shape[0]
        if N_i == 0:
            z_clusters_intrinsic.append(torch.zeros((0, d), device=device))
            continue

        std_U_i = U_i.std(dim=0, keepdim=True).to(device) if N_i > 1 else torch.ones((1, d), device=device)
        Z_raw_i = torch.randn((N_i, d), device=device) * std_U_i

        cost_matrix_i = torch.cdist(Z_raw_i, U_i, p=2)**2
        plan_i = ot.emd(np.ones(N_i)/N_i, np.ones(N_i)/N_i, cost_matrix_i.cpu().numpy())
        z_clusters_intrinsic.append(Z_raw_i[np.argmax(plan_i, axis=0)])

    sys.stdout.write("\n")
    torch.save([z.cpu() for z in z_clusters_intrinsic], os.path.join(args.data_dir, "z_clusters_intrinsic.pt"))

    # =========================================================
    # PILLAR 2: WAVELET MAP CALIBRATION (With Chart Advancement)
    # =========================================================
    print(f"Calibrating {m} chart-decoupled Besov wavelet feature maps at \beta = {besov_beta:.2f}...")
    chart_wavelet_maps = []
    raw_p_trunc = config['features']['p_trunc']
    is_auto_p = str(raw_p_trunc).strip().lower() in ['auto', 'none', '0']

    for i in range(m):
        sys.stdout.write(f"\r[RKHS Calibration] Calibrated chart {i + 1}/{m} ({(i + 1) / m * 100:.1f}%)")
        sys.stdout.flush()

        U_i = chart_intrinsic_coords[i]
        N_i = U_i.shape[0]
        
        if N_i <= 1:
            chart_wavelet_maps.append(None)
            continue
            
        if is_auto_p:
            p_calc = int(N_i * math.log(N_i)) if N_i > 1 else 128
            p_trunc_i = (max(128, min(p_calc, 1024)) // 4) * 4
        else:
            p_trunc_i = (int(float(raw_p_trunc)) // 4) * 4
            
        model_i = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_trunc_i).to(device)
        model_i.calibrate(U_i, base_beta=besov_beta)
        model_i.eval()
        chart_wavelet_maps.append(model_i.state_dict())

    sys.stdout.write("\n")
    torch.save(chart_wavelet_maps, os.path.join(args.data_dir, "wavelet_maps_decoupled.pt"))

    # =========================================================
    # PILLAR 3: VECTOR FIELD REGRESSION (With Step & Chart Advancement)
    # =========================================================
    s = torch.linspace(0, 1.0, time_steps)
    time_grid = torch.sinh(s * 2.0) / torch.sinh(torch.tensor(2.0))
    all_etas = []

    print(f"Regressing {m} Besov-regularized vector fields across {time_steps} steps...")
    for step in range(time_steps):
        t_val = time_grid[step].item()
        eta_t_local = []
        
        for i in range(m):
            # Granular step and chart cross-advancement tracker
            sys.stdout.write(f"\r[Drift Regression] Step {step + 1}/{time_steps} | Solved chart {i + 1}/{m} ({(step * m + i + 1) / (time_steps * m) * 100:.1f}%)")
            sys.stdout.flush()

            U_i = chart_intrinsic_coords[i]
            Z_i = z_clusters_intrinsic[i]
            
            if U_i.shape[0] <= 1 or chart_wavelet_maps[i] is None:
                eta_t_local.append(torch.zeros((1, d), device=device))
                continue
                
            p_i = chart_wavelet_maps[i]['omega'].shape[0]
            model_i = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_i).to(device)
            model_i.load_state_dict(chart_wavelet_maps[i])
            
            eta_t_local.append(solve_local_system(
                features=model_i((1.0 - t_val) * Z_i + t_val * U_i),
                target_velocities=U_i - Z_i,
                rkhs_penalty=model_i.get_rkhs_penalty()
            ))

        all_etas.append(eta_t_local)

    sys.stdout.write("\n[Phase 2 Complete] All local velocity fields successfully serialized.\n")
    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"))

if __name__ == "__main__": main()