import torch
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.wavelet_map import TruncatedBesovWaveletMap, compute_feature_gradients
from src.local_linear_regression import solve_local_system

def sinkhorn_ot(X: torch.Tensor, Y: torch.Tensor, epsilon: float = 0.05, iterations: int = 100) -> torch.Tensor:
    """
    Log-domain Entropic Regularized Optimal Transport.
    Returns the permutation indices mapping Y to X.
    Executes entirely on GPU with O(N^2) complexity.
    """
    N = X.size(0)
    C = torch.cdist(X, Y, p=2) ** 2
    
    # Log-domain initialization
    log_K = -C / epsilon
    log_u = torch.zeros(N, device=X.device)
    log_v = torch.zeros(N, device=Y.device)
    
    for _ in range(iterations):
        # Update v
        log_v = -torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        # Update u
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        
    log_P = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    
    # Extract hard assignment via argmax
    return torch.argmax(P, dim=1)

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--ambient_dim", type=int, default=16)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=64)
    parser.add_argument("--time_steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.load(os.path.join(args.data_dir, "data.pt")).to(device)
    labels = torch.load(os.path.join(args.data_dir, "labels.pt")).to(device)
    num_charts = int(labels.max().item() + 1)

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc).to(device)
    model.eval()

    print("Computing Entropic Optimal Transport (Sinkhorn) matching...")
    Z_raw = torch.randn_like(data)
    
    # GPU-accelerated matching
    col_ind = sinkhorn_ot(Z_raw, data, epsilon=0.05, iterations=100)
    
    Z = torch.zeros_like(Z_raw)
    Z[col_ind] = Z_raw # Permute Z to match data
    
    z_clusters = [Z[labels == i].cpu() for i in range(num_charts)]
    torch.save(z_clusters, os.path.join(args.data_dir, "z_clusters.pt"))

    time_grid = torch.linspace(0, 1.0, args.time_steps)
    all_etas = []
    print(f"Solving {num_charts} local systems across {args.time_steps} time steps...")

    for step, t in enumerate(time_grid):
        eta_t_local = []
        I_t = (1.0 - t.item()) * Z + t.item() * data
        dot_I_t = data - Z
        
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