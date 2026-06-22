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

    # ---------------------------------------------------------
    # AUTOMATED MINIMAX PARAMETER DETERMINATION
    # ---------------------------------------------------------
    raw_p_trunc = config['features']['p_trunc']
    if str(raw_p_trunc).strip().lower() in ['auto', 'none', '0']:
        max_N_i = max([U.shape[0] for U in chart_intrinsic_coords])
        if max_N_i > 1:
            p_trunc_calc = int(max_N_i * math.log(max_N_i))
        else:
            p_trunc_calc = 128
        p_trunc = max(128, min(p_trunc_calc, 5000))
        p_trunc = (p_trunc // 4) * 4
        print(f"[DEBUG] Auto p_trunc (P) = {p_trunc} (Based on max patch size {max_N_i})")
    else:
        p_trunc = int(float(raw_p_trunc))

    raw_time_steps = config['integration']['time_steps']
    if str(raw_time_steps).strip().lower() in ['auto', 'none', '0']:
        beta = 1.5
        exponent = beta / (2.0 * beta + float(d))
        t_calc = int(15.0 * math.pow(N, exponent))
        time_steps = max(50, min(t_calc, 2000))*2
        print(f"[DEBUG] Auto time_steps = {time_steps} (Based on N={N}, d={d}, beta={beta})")
    else:
        time_steps = int(float(raw_time_steps))
    # ---------------------------------------------------------

    print(f"Executing Covariance-Matched Exact Optimal Transport in R^{d}...")
    z_clusters_intrinsic = []
    torch.manual_seed(42)

    for i in range(m):
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

    torch.save([z.cpu() for z in z_clusters_intrinsic], os.path.join(args.data_dir, "z_clusters_intrinsic.pt"))

    model = TruncatedBesovWaveletMap(ambient_dim=d, intrinsic_dim=d, p_truncation=p_trunc).to(device)
    model.calibrate(torch.cat(chart_intrinsic_coords, dim=0))
    model.eval()

    torch.save(model.state_dict(), os.path.join(args.data_dir, "wavelet_map.pt"))

    s = torch.linspace(0, 1.0, time_steps)
    time_grid = torch.sinh(s * 2.0) / torch.sinh(torch.tensor(2.0))
    all_etas = []

    print(f"Regressing {m} Besov-regularized vector fields across {time_steps} steps...")
    for step in range(time_steps):
        t_val = time_grid[step].item()
        eta_t_local = []
        for i in range(m):
            U_i = chart_intrinsic_coords[i]
            Z_i = z_clusters_intrinsic[i]
            if U_i.shape[0] == 0:
                eta_t_local.append(torch.zeros((p_trunc, d), device=device))
                continue

            eta_t_local.append(solve_local_system(
                features=model((1.0 - t_val) * Z_i + t_val * U_i),
                target_velocities=U_i - Z_i,
                rkhs_penalty=model.get_rkhs_penalty()
            ))

        all_etas.append(eta_t_local)
        if (step + 1) % 25 == 0: print(f" Solved step {step + 1}/{time_steps}")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas_intrinsic.pt"))

if __name__ == "__main__": main()