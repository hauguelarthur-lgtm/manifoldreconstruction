import torch
import os
import sys
import argparse
import ot
from scipy.spatial.distance import cdist
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.wavelet_map import TruncatedBesovWaveletMap, compute_feature_gradients
from src.local_linear_regression import solve_local_system
from src.projector import GlobalSubspaceProjector

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.load(os.path.join(args.data_dir, "data.pt")).to(device)
    labels = torch.load(os.path.join(args.data_dir, "labels.pt")).to(device)
    num_charts = int(labels.max().item() + 1)

    # 1. Execute SVD Global Subspace Truncation
    projector = GlobalSubspaceProjector(variance_threshold=0.999)
    data_k = projector.fit_transform(data)
    k = projector.k
    
    torch.save(projector, os.path.join(args.data_dir, "projector.pt"))
    torch.save(data_k, os.path.join(args.data_dir, "data_k.pt"))

    # 2. Recompute Euclidean barycenters and Precision matrices strictly in R^k
    centers_k = []
    precisions_k = []
    for i in range(num_charts):
        mask = (labels == i)
        chart_data = data_k[mask]
        
        mu = chart_data.mean(dim=0)
        centers_k.append(mu)
        
        centered = chart_data - mu
        dof = max(chart_data.size(0) - 1, 1)
        cov = torch.matmul(centered.T, centered) / dof
        
        # Pseudo-inverse to isolate local tangent spaces within R^k
        precisions_k.append(torch.linalg.pinv(cov, rcond=1e-3))

    torch.save(torch.stack(centers_k), os.path.join(args.data_dir, "cluster_centers_k.pt"))
    torch.save(torch.stack(precisions_k), os.path.join(args.data_dir, "cluster_precisions_k.pt"))

    # 3. Initialize model in the reduced dimension k
    model = TruncatedBesovWaveletMap(k, args.intrinsic_dim, args.p_trunc).to(device)
    print("Calibrating Random Fourier Features via median distance heuristic...")
    model.calibrate(data_k)
    model.eval()

    # 4. Compute Exact Optimal Transport in R^k
    print(f"Computing Exact L2 Optimal Transport matching (POT EMD) in R^{k}...")
    Z_raw = torch.randn_like(data_k)
    
    N = Z_raw.shape[0]
    a = np.ones(N) / N
    b = np.ones(N) / N
    
    cost_matrix = cdist(Z_raw.cpu().numpy(), data_k.cpu().numpy(), metric='sqeuclidean')
    T = ot.emd(a, b, cost_matrix)
    
    col_ind = np.argmax(T, axis=1)
    row_ind = np.arange(N)
    
    Z = torch.zeros_like(Z_raw)
    Z[col_ind] = Z_raw[row_ind]
    
    z_clusters = [Z[labels == i].cpu() for i in range(num_charts)]
    torch.save(z_clusters, os.path.join(args.data_dir, "z_clusters.pt"))

    time_grid = torch.linspace(0, 1.0 - 1e-5, args.time_steps)
    print(f"Solving {num_charts} local systems batched across {args.time_steps} time steps...")
    
    t_tensor = time_grid.view(-1, 1, 1).to(device)
    I_t_batch = (1.0 - t_tensor) * Z.unsqueeze(0) + t_tensor * data_k.unsqueeze(0)
    dot_I_t = data_k - Z
    
    all_etas = []
    
    for step in range(args.time_steps):
        eta_t_local = []
        I_t = I_t_batch[step]
        
        for i in range(num_charts):
            chart_mask = (labels == i)
            I_t_i = I_t[chart_mask]
            dot_I_t_i = dot_I_t[chart_mask]
            
            if I_t_i.size(0) == 0:
                eta_t_local.append(torch.zeros(args.p_trunc, device=device))
                continue
                
            feature_grads_i = compute_feature_gradients(model, I_t_i)
            eta = solve_local_system(feature_grads_i, dot_I_t_i)
            eta_t_local.append(eta)
            
        all_etas.append(eta_t_local)
        if (step + 1) % 10 == 0:
            print(f"Solved step {step + 1}/{args.time_steps}")

    torch.save(all_etas, os.path.join(args.data_dir, "precomputed_etas.pt"))

if __name__ == "__main__":
    main()