import os
import sys
import torch
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ensure project paths are resolved
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None
try:
    from scripts.verificationsconditions import execute_verification_suite, load_artifacts
except ImportError:
    print("Error: Must ensure 06_verify_conditions.py is accessible for the verification logic.")
    sys.exit(1)

# Import your manifold construction module
# Replace 'manifoldclustering' with the exact name of your python file/class
from src.manifoldclustering import construct_whitney_atlas 

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
    
    # 1. Minimax Dilation Penalty (Heavy penalty for overriding statistical bandwidth)
    # Range [0.0, 1.0], scaled to [0, 1000]
    dilation_ratio = report["5.1_DualRadius_Dominance"]["value"]
    score += dilation_ratio * 1000.0
    
    # 2. Barycentric Degeneracy Penalty (Heavy penalty if centers collapse below 0.25 * delta_n)
    separation = report["1.2_Intrinsic_Geodesic_Separation"]["value"]
    target_degeneracy = report["1.2_Intrinsic_Geodesic_Separation"].get("target_degeneracy_limit", 1e-3)
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

def execute_grid_search(data_tensor: torch.Tensor, output_dir: str):
    """
    Executes the combinatorial search space and serializes the validation metrics.
    """
    # Define the discrete search space
    grid_params = {
        'beta': [1.0, 1.5, 2.0],
        'bandwidth_multiplier': [0.5, 1.0, 1.5, 2.5],
        'tau_reach_limit': [1.5, 2.5, 4.0],
        'local_scale_neighbor': [3, 5, 8],
        'k_max': [30, 50]
    }
    
    keys = grid_params.keys()
    combinations = list(itertools.product(*(grid_params[k] for k in keys)))
    
    print(f"Executing automated hyperparameter grid search across {len(combinations)} configurations...")
    
    results_log = []
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Evaluation {idx+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        try:
            # 1. Initialize and execute your specific pipeline
            # Note: You must map these parameters to your exact class initialization
            constructor = construct_whitney_atlas(
                beta=params['beta'],
                bandwidth_multiplier=params['bandwidth_multiplier'],
                tau_reach_limit=params['tau_reach_limit'],
                local_scale_neighbor=params['local_scale_neighbor'],
                k_max=params['k_max']
            )
            
            # Execute the clustering and projection
            constructor.fit(data_tensor)
            constructor.save_artifacts(output_dir)
            
            # 2. Load the freshly generated artifacts
            data, mask, coords, atlas, indices = load_artifacts(output_dir)
            
            # 3. Evaluate the strict geometric conditions
            report = execute_verification_suite(data, mask, coords, atlas, indices)
            
            # 4. Compute unified fitness scalar
            fitness = compute_fitness_score(report)
            
            # 5. Log metrics
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

    # Serialize results to a structured DataFrame
    df_results = pd.DataFrame(results_log)
    df_results = df_results.sort_values(by='fitness_score', ascending=True)
    
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"Grid search complete. Full topological analysis serialized to: {csv_path}")
    print("Optimal Hyperparameter Configuration Found:")
    print(df_results.iloc[0].to_dict())

if __name__ == "__main__":
    # Ensure this path maps to your standardized empirical tensor
    data_path = os.path.join(project_root, "data", "processed", "data.pt")
    out_dir = os.path.join(project_root, "data", "processed")
    
    if not os.path.exists(data_path):
        print(f"Target dataset tensor not found at {data_path}. Generate geometry first.")
        sys.exit(1)
        
    dataset = torch.load(data_path)
    execute_grid_search(dataset, out_dir)