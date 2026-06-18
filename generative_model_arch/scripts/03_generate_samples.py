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
from src.diffusion import compute_optimal_diffusion

def compute_drift(X: torch.Tensor, model: torch.nn.Module, etas_t: list, chart_assignments: torch.Tensor, num_charts: int) -> torch.Tensor:
    """Computes the drift strictly on necessary particles, avoiding redundant memory allocations."""
    b_t = torch.zeros_like(X)
    for i in range(num_charts):
        mask = (chart_assignments == i)
        if not mask.any(): continue
            
        X_i = X[mask]
        eta_i = etas_t[i].to(X.device)
        
        # O(1) memory optimization: Compute gradients ONLY for particles in chart i
        grads_i = compute_feature_gradients(model, X_i) 
        b_t[mask] = torch.matmul(grads_i.transpose(1, 2), eta_i.unsqueeze(1)).squeeze(-1)
        
    return b_t

def generate_samples(model: torch.nn.Module,
                     precomputed_etas: list, 
                     z_clusters: list,
                     chart_assignments: torch.Tensor,
                     num_samples: int, 
                     ambient_dim: int, 
                     num_time_steps: int = 100,
                     device: torch.device = torch.device('cpu'),
                     ode_mode: bool = False) -> torch.Tensor:
    
    X_t = torch.zeros(num_samples, ambient_dim, device=device)
    num_charts = len(z_clusters)
    
    for i in range(num_charts):
        mask = (chart_assignments == i)
        n_samples_i = mask.sum().item()
        if n_samples_i == 0: continue
            
        Z_i = z_clusters[i].to(device)
        mu_i = Z_i.mean(dim=0)
        cov_i = torch.cov(Z_i.T) + torch.eye(ambient_dim, device=device) * 1e-4
        dist = torch.distributions.MultivariateNormal(mu_i, cov_i)
        X_t[mask] = dist.sample((n_samples_i,))

    dt = 1.0 / num_time_steps
    time_grid = torch.linspace(0, 1.0 - dt, num_time_steps)
    model.eval()
    
    max_drift = 10.0

    for step, t in enumerate(time_grid):
        X_t = X_t.detach()
        etas_t = precomputed_etas[step] 
        
        # 1. Compute Drift k1 at X_t
        b_t = compute_drift(X_t, model, etas_t, chart_assignments, num_charts)
        
        # Clipping logic applies to both integration modes
        drift_norms = torch.norm(b_t, dim=1, keepdim=True)
        b_t = b_t * torch.clamp(max_drift / (drift_norms + 1e-8), max=1.0)
        
        if step % 10 == 0:
            print(f"Step {step} | Mean L2 Drift: {torch.norm(b_t, dim=1).mean().item():.6f}")

        if ode_mode:
            # 2. Heun's Method (RK2) for ODE
            # Predictor step
            X_pred = X_t + b_t * dt 
            
            # Read forward velocity if not at terminal step
            if step < len(time_grid) - 1:
                etas_t_next = precomputed_etas[step + 1]
                b_t_next = compute_drift(X_pred, model, etas_t_next, chart_assignments, num_charts)
                drift_norms_next = torch.norm(b_t_next, dim=1, keepdim=True)
                b_t_next = b_t_next * torch.clamp(max_drift / (drift_norms_next + 1e-8), max=1.0)
                
                # Corrector step (Average of k1 and k2)
                X_t = X_t + 0.5 * (b_t + b_t_next) * dt
            else:
                X_t = X_pred
        else:
            # Standard Euler-Maruyama for SDE
            D_t_star = compute_optimal_diffusion(t.item())
            dW = torch.randn_like(X_t) * torch.sqrt(torch.tensor(dt, device=device))
            X_t = X_t + b_t * dt + torch.sqrt(torch.tensor(2 * D_t_star, device=device)) * dW
            
    return X_t.detach()

def main():
    default_data_dir = os.path.join(project_root, "data", "processed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--ambient_dim", type=int, default=16)
    parser.add_argument("--intrinsic_dim", type=int, default=4)
    parser.add_argument("--p_trunc", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--ode_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = torch.load(os.path.join(args.data_dir, "labels.pt"), map_location=device)
    precomputed_etas = torch.load(os.path.join(args.data_dir, "precomputed_etas.pt"), map_location=device)
    z_clusters = torch.load(os.path.join(args.data_dir, "z_clusters.pt"), map_location=device)

    num_charts = int(labels.max().item() + 1)
    chart_counts = torch.bincount(labels, minlength=num_charts).float()
    chart_probs = chart_counts / chart_counts.sum()
    chart_assignments = torch.multinomial(chart_probs, args.num_samples, replacement=True).to(device)

    model = TruncatedBesovWaveletMap(args.ambient_dim, args.intrinsic_dim, args.p_trunc).to(device)
    
    generated_data = generate_samples(
        model=model,
        precomputed_etas=precomputed_etas,
        z_clusters=z_clusters,
        chart_assignments=chart_assignments,
        num_samples=args.num_samples,
        ambient_dim=args.ambient_dim,
        num_time_steps=args.time_steps,
        device=device,
        ode_mode=args.ode_mode
    )

    torch.save(generated_data, os.path.join(args.data_dir, "generated_samples.pt"))

if __name__ == "__main__":
    main()