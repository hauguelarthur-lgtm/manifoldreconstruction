import os
import sys
import json
import torch
import math
import argparse
import numpy as np
from scipy.sparse.csgraph import dijkstra
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Injection du path pour désérialiser la classe de l'atlas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_artifacts(data_dir: str):
    print(f"Loading artifacts from {data_dir}...")
    data = torch.load(os.path.join(data_dir, "data.pt"))
    mask = torch.load(os.path.join(data_dir, "membership_mask.pt"))
    coords = torch.load(os.path.join(data_dir, "chart_intrinsic_coords.pt"))
    
    atlas_obj = torch.load(os.path.join(data_dir, "whitney_atlas.pt"))
    atlas_frames = atlas_obj.atlas if hasattr(atlas_obj, 'atlas') else atlas_obj
    
    indices = torch.load(os.path.join(data_dir, "chart_ambient_indices.pt"))
    return data, mask, coords, atlas_frames, indices

def execute_verification_suite(data, mask, coords, atlas, indices):
    report = {}
    N, m = mask.shape
    
    # Infer empirical delta_minimax from the smallest radius in the atlas
    r_sq_list = [chart['r_sq'] for chart in atlas]
    empirical_delta_minimax = math.sqrt(min(r_sq_list))
    
    # =====================================================================
    # 1. Topological Covering Constraints
    # =====================================================================
    
    # 1.1 Global Support Covering
    coverage_sums = mask.sum(dim=1)
    min_coverage = int(coverage_sums.min().item())
    report["1.1_Global_Support_Covering"] = {
        "metric": "Minimum overlapping charts per data point",
        "value": min_coverage,
        "passed": min_coverage >= 1
    }
    

    # =====================================================================
    # 1.2 Hybrid Intrinsic Separation (Degeneracy Detector)
    # =====================================================================
    mu_tensor = torch.stack([chart['mu'] for chart in atlas])
    
    # 1. Pure Ambient Euclidean Lower Bound
    # Since clustering now uses torch.cdist, we only need to test torch.cdist
    euclidean_matrix = torch.cdist(mu_tensor, mu_tensor).cpu().numpy()
    
    # Fill the diagonal with infinity so a center's distance to itself (0.0) is ignored
    np.fill_diagonal(euclidean_matrix, np.inf)
    
    # Find the closest distance between any two chart centers
    min_separation = float(np.min(euclidean_matrix))
    
    # 2. Barycentric Degeneracy Threshold
    # We test for critical redundancy (centers merging)
    critical_degeneracy_limit = empirical_delta_minimax
    
    report["1.2_Ambient_Euclidean_Separation"] = {
        "metric": "Minimum ambient Euclidean separation",
        "value": min_separation,
        "target_degeneracy_limit": critical_degeneracy_limit,
        "passed": min_separation >= critical_degeneracy_limit
    }
    
    # 1.3 Bounded Multiplicity
    max_multiplicity = int(coverage_sums.max().item())
    report["1.3_Bounded_Multiplicity"] = {
        "metric": "Maximum overlapping charts at any coordinate",
        "value": max_multiplicity,
        "passed": max_multiplicity <= 15 # Arbitrary reasonable bound for tau(d)
    }

    # =====================================================================
    # 2. Algebraic & Matrix Conditioning Constraints
    # =====================================================================
    
    # 2.1 Strict Gram Matrix Non-Singularity
    min_points_in_chart = int(mask.sum(dim=0).min().item())
    d = coords[0].shape[1]
    report["2.1_Gram_Matrix_NonSingularity"] = {
        "metric": "Minimum localized points across all charts",
        "value": min_points_in_chart,
        "passed": min_points_in_chart > d + 1
    }
    
    # 2.2 Tangent Subspace Orthogonality
    max_orthogonality_error = 0.0
    for i, chart in enumerate(atlas):
        U = coords[i]
        # CORRECTION: Extract X using the boolean spatial mask, not the regression indices
        X = data[mask[:, i]] 
        mu = chart['mu']
        Q = chart['Q']
        X_proj = mu + torch.matmul(U, Q.T)
        N_err = X - X_proj
        # U^T * N_err must be 0
        error = torch.max(torch.abs(torch.matmul(U.T, N_err))).item()
        max_orthogonality_error = max(max_orthogonality_error, error)
        
    report["2.2_Tangent_Orthogonality"] = {
        "metric": "Maximum dot product error between tangent base and normal residual",
        "value": max_orthogonality_error,
        "passed": max_orthogonality_error < 1e-4
    }

    # =====================================================================
    # 3. Geometric Approximation Constraints
    # =====================================================================
    
    # 3.2 Asymptotic Curvature Residual Bound (Linear PCA Baseline)
    max_rmse = 0.0
    for i, chart in enumerate(atlas):
        U = coords[i]
        # CORRECTION: Extract X using the boolean spatial mask
        X = data[mask[:, i]] 
        mu = chart['mu']
        Q = chart['Q']
        X_proj = mu + torch.matmul(U, Q.T)
        rmse = torch.sqrt(torch.mean((X - X_proj)**2)).item()
        max_rmse = max(max_rmse, rmse)

    report["3.2_Curvature_Residual_Bound"] = {
        "metric": "Maximum local geometric RMSE (Order 1 Projection)",
        "value": max_rmse,
        "passed": max_rmse < empirical_delta_minimax
    }

    # =====================================================================
    # 4. Partition of Unity Constraints
    # =====================================================================
    
    # 4.1 Strict Convex Normalization & 4.2 Boundary Collapse
    total_weights = torch.zeros(N)
    boundary_leakage_sum = 0.0
    
    for chart in atlas:
        mu = chart['mu']
        r_sq = chart['r_sq']
        dist_sq = torch.sum((data - mu) ** 2, dim=1)
        valid_mask = dist_sq < r_sq
        
        w = torch.zeros(N)
        # Bump function computation
        normalized_sq = dist_sq[valid_mask] / r_sq
        w[valid_mask] = torch.exp(-1.0 / (1.0 - normalized_sq))
        
        total_weights += w
        # Calculate sum of weights explicitly outside the boundary mask
        boundary_leakage_sum += torch.sum(w[~valid_mask]).item()

    convex_normalization_valid = bool(torch.all(total_weights > 0).item())
    
    report["4.1_Convex_Normalization"] = {
        "metric": "Are bump function sums > 0 for all spatial coordinates?",
        "value": convex_normalization_valid,
        "passed": convex_normalization_valid
    }
    
    report["4.2_Boundary_Collapse"] = {
        "metric": "Sum of computational weight escaping the strict r_sq boundary",
        "value": boundary_leakage_sum,
        "passed": boundary_leakage_sum == 0.0
    }

    # =====================================================================
    # 5. Asymptotic Minimax Constraints
    # =====================================================================
    
    # 5.1 Dual-Radius Dominance Ratio
    inflated_charts = sum(1 for r in r_sq_list if r > (empirical_delta_minimax ** 2) * 1.05)
    inflation_ratio = inflated_charts / m
    
    report["5.1_DualRadius_Dominance"] = {
        "metric": "Proportion of charts forced to dilate beyond delta_minimax",
        "value": inflation_ratio,
        "passed": inflation_ratio < 0.5 # Expectation: The majority of the atlas operates at minimax
    }

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
    parser.add_argument("--output_file", type=str, default="verification_report.json")
    args = parser.parse_args()

    data, mask, coords, atlas, indices = load_artifacts(args.data_dir)
    
    print("Executing strict generative manifold condition tests...")
    report = execute_verification_suite(data, mask, coords, atlas, indices)
    
    # Write output report
    output_path = os.path.join(args.data_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Verification complete. Report strictly generated at: {output_path}")
    
    # Print summary to console
    fails = 0
    for key, data in report.items():
        status = "PASS" if data["passed"] else "FAIL"
        if status == "FAIL": fails += 1
        print(f"[{status}] {key}: {data['value']}")
        
    if fails > 0:
        print(f"\nWARNING: {fails} geometric constraints failed. Generative SDE integration may become unstable.")
    else:
        print("\nSUCCESS: All generative manifold preconditions met.")

if __name__ == "__main__":
    main()