import os
import sys
import torch
import argparse
import itertools
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any

# 1. Strict Path Resolution for the generative_model_arch repository structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Import the verification suite and functional entry point
try:
    from scripts.verificationsconditions import execute_verification_suite, load_artifacts
except ImportError as e:
    print(f"Error: Must ensure scripts/verificationsconditions.py is accessible. Details: {e}")
    sys.exit(1)

try:
    from src.manifoldclustering import construct_whitney_atlas
except ImportError as e:
    print(f"Error: Could not import construct_whitney_atlas from src.manifoldclustering. Details: {e}")
    sys.exit(1)

def compute_fitness_score(report: Dict[str, Any]) -> float:
    """
    Computes a penalized scalar fitness score based on geometric constraints.
    Lower score is strictly better. A score of infinity implies a fatal topological collapse.
    """
    score = 0.0
    
    # Hard Failures (Fatal Topology or Algebraic Singularity)
    if not report["1.1_Global_Support_Covering"]["passed"]: return float('inf')
    if not report["2.1_Gram_Matrix_NonSingularity"]["passed"]: return float('inf')
    if not report["4.1_Convex_Normalization"]["passed"]: return float('inf')
    
    # 1. Minimax Dilation Penalty
    dilation_ratio = report["5.1_DualRadius_Dominance"]["value"]
    score += dilation_ratio * 1000.0
    
    # 2. Barycentric Degeneracy Penalty 
    separation = report["1.2_Intrinsic_Geodesic_Separation"]["value"]
    target_degeneracy = report.get("1.2_Intrinsic_Geodesic_Separation", {}).get("target_degeneracy_limit", 1e-3)
    if separation < target_degeneracy:
        score += 500.0 * (1.0 - (separation / target_degeneracy))
        
    # 3. Multiplicity Penalty (Optimizes for sparse Partition of Unity)
    multiplicity = report["1.3_Bounded_Multiplicity"]["value"]
    if multiplicity > 15:
        score += (multiplicity - 15) * 10.0
        
    # 4. Geometric Precision (Minimizes Order 1 Projection RMSE)
    rmse = report["3.2_Curvature_Residual_Bound"]["value"]
    score += rmse * 100.0
    
    return score

def execute_grid_search(data_tensor: torch.Tensor, output_dir: str, config_dir: str, intrinsic_d: int):
    """
    Executes the combinatorial search space utilizing the functional construct_whitney_atlas entry point.
    """
    grid_params = {
        'beta': [1.0, 1.5, 2.0],
        'bandwidth_multiplier': [0.5, 1.0, 1.5, 2.5],
        'tau_reach_limit': [1.5, 2.5, 4.0],
        'local_scale_neighbor': [3, 5, 8],
        'k_max': [30, 50]
    }
    
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*(grid_params[k] for k in keys)))
    
    print(f"Executing automated hyperparameter grid search across {len(combinations)} configurations...")
    print(f"Intrinsic dimension strictly enforced: d={intrinsic_d}")
    
    results_log = []
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Evaluation {idx+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        # Construct the empirical config dictionary block for functional injection
        empirical_config = {
            'beta': params['beta'],
            'bandwidth_multiplier': params['bandwidth_multiplier'],
            'tau_reach_limit': params['tau_reach_limit'],
            'local_scale_neighbor': params['local_scale_neighbor'],
            'k_max': params['k_max']
        }
        
        try:
            # 1. Execute the functional pipeline utilizing dictionary unpacking for the empirical config
            atlas, mask, coords, indices = construct_whitney_atlas(
                data_tensor,
                intrinsic_dim=d,
                empirical_config=empirical_config
            )
            
            # 2. Serialize artifacts directly to the output directory
            torch.save(data_tensor, os.path.join(output_dir, "data.pt"))
            torch.save(mask, os.path.join(output_dir, "membership_mask.pt"))
            torch.save(coords, os.path.join(output_dir, "chart_intrinsic_coords.pt"))
            torch.save(atlas, os.path.join(output_dir, "whitney_atlas.pt"))
            torch.save(indices, os.path.join(output_dir, "chart_ambient_indices.pt"))
            
            # 3. Load the freshly generated artifacts utilizing verificationsconditions.py
            data, loaded_mask, loaded_coords, loaded_atlas, loaded_indices = load_artifacts(output_dir)
            
            # 4. Evaluate the strict geometric conditions
            report = execute_verification_suite(data, loaded_mask, loaded_coords, loaded_atlas, loaded_indices)
            
            # 5. Compute unified fitness scalar
            fitness = compute_fitness_score(report)
            
            # 6. Log metrics
            log_entry = {**params}
            log_entry['fitness_score'] = fitness
            log_entry['dilation_ratio'] = report["5.1_DualRadius_Dominance"]["value"]
            log_entry['min_separation'] = report["1.2_Intrinsic_Geodesic_Separation"]["value"]
            log_entry['max_multiplicity'] = report["1.3_Bounded_Multiplicity"]["value"]
            log_entry['max_rmse'] = report["3.2_Curvature_Residual_Bound"]["value"]
            log_entry['status'] = "SUCCESS" if fitness != float('inf') else "FATAL_COLLAPSE"
            
            results_log.append(log_entry)
            print(f"Score: {fitness:.4f} | Status: {log_entry['status']}")
            
        except Exception as e:
            print(f"Execution failed for configuration {params}: {str(e)}")
            log_entry = {**params, 'fitness_score': float('inf'), 'status': f"ERROR: {str(e)}"}
            results_log.append(log_entry)

    # 7. Serialize results to a structured DataFrame
    df_results = pd.DataFrame(results_log)
    df_results = df_results.sort_values(by='fitness_score', ascending=True)
    
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    # 8. Construct and strictly serialize the optimal empirical config file
    best_params = df_results.iloc[0].to_dict()
    optimal_empirical_config = {
        'empirical_geometry': {
            'N': int(data_tensor.shape[0]),
            'ambient_p': int(data_tensor.shape[1]),
            'intrinsic_d': int(intrinsic_d)
        },
        'optimal_hyperparameters': {
            'beta': float(best_params['beta']),
            'bandwidth_multiplier': float(best_params['bandwidth_multiplier']),
            'tau_reach_limit': float(best_params['tau_reach_limit']),
            'local_scale_neighbor': int(best_params['local_scale_neighbor']),
            'k_max': int(best_params['k_max'])
        },
        'validation_metrics': {
            'fitness_score': float(best_params['fitness_score']),
            'dilation_ratio': float(best_params['dilation_ratio']),
            'min_separation': float(best_params['min_separation']),
            'max_multiplicity': int(best_params['max_multiplicity']),
            'max_rmse': float(best_params['max_rmse'])
        }
    }
    
    config_file_path = os.path.join(config_dir, "optimal_empirical_config.yaml")
    with open(config_file_path, 'w') as f:
        yaml.dump(optimal_empirical_config, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*50)
    print(f"Grid search complete. Full topological analysis serialized to: {csv_path}")
    print(f"Optimal empirical config strictly mapped and saved to: {config_file_path}")
    print(df_results.iloc[0].to_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Grid Search for Whitney Atlas Construction.")
    parser.add_argument("--d", type=int, required=True, help="Strict intrinsic dimension d of the empirical manifold.")
    args = parser.parse_args()

    # Standardized empirical tensor path resolution
    data_path = os.path.join(project_root, "data", "processed", "data.pt")
    out_dir = os.path.join(project_root, "data", "processed")
    config_dir = os.path.join(project_root, "configs")
    
    if not os.path.exists(data_path):
        print(f"Target dataset tensor not found at {data_path}. Generate geometry first.")
        sys.exit(1)
        
    # Force creation of directories if missing
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
        
    dataset = torch.load(data_path)
    execute_grid_search(dataset, out_dir, config_dir, args.d)