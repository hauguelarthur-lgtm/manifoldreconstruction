import torch
import os
import sys
import argparse
import yaml
import itertools
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.manifold_extraction.manifoldclusteringOriginal import construct_whitney_atlas, EmpiricalConfig, get_poly_features, WhitneyPartitionOfUnity

# Assumes score_function is imported from your evaluation module
from src.evaluation.tests_metric import test_metrics, evaluate_reconstruction_mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(project_root, "data", "raw", "dataset.pt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config = yaml.safe_load(f)
        
    d = int(config['manifold']['intrinsic_dim'])
    p = int(config['manifold']['ambient_dim'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_ambient = torch.load(args.data_path, map_location=device)

    # 1. DEFINE GRID SEARCH PARAMETER SPACE
    # Overrides the single dictionary assignment to evaluate multiple hypotheses
    grid_space = {
        'volume_scale': np.arange(1, 20, 0.5),
        'C_overlap': [1],
        'beta': np.arange(0.5, 8.5, 1)
    }

    keys = list(grid_space.keys())
    values = list(grid_space.values())
    
    best_score = float('inf')
    best_params = None
    best_artifacts = None

    total_combinations = len(list(itertools.product(*values)))
    print(f"Starting Grid Search over {total_combinations} combinations...")

    # 2. EXECUTE OPTIMIZATION LOOP
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        
        # Initialize the Empirical tuning configuration dynamically[cite: 2]
        empirical_params = EmpiricalConfig(
            volume_scale=float(param_dict['volume_scale']),
            C_overlap=float(param_dict['C_overlap']),
            beta=float(param_dict['beta'])
        )

        try:
            print(f"parameters: {param_dict}")
            # Execute radius-driven atlas construction[cite: 2]
            (membership_mask, chart_intrinsic_coords, whitney_atlas, chart_ambient_indices, chart_centers_indices) = construct_whitney_atlas(
                data=data_ambient,
                intrinsic_dim=d,
                empirical_config=empirical_params
            )

            # Evaluate the instantiated artifacts
            current_score = test_metrics(
                whitney_atlas,
                chart_centers_indices,
                membership_mask,
                data_ambient,
                d=d,
                p=p,
                beta= empirical_params.beta
            )

            # Update optimal parameters
            if current_score < best_score:
                best_score = current_score
                print(best_score)
                best_params = param_dict
                best_artifacts = (membership_mask, chart_intrinsic_coords, whitney_atlas, chart_ambient_indices, chart_centers_indices)
                print(f"New best score: {best_score} via parameters: {best_params}")

        except Exception as e:
            # Trap linear algebra exceptions (e.g., torch._C._LinAlgError) caused by poor Tikhonov regularization bounds
            pass

    if best_score == 0:
        raise RuntimeError("Grid search failed to resolve any mathematically viable configurations.")

    # Unpack optimal artifacts
    membership_mask, chart_intrinsic_coords, whitney_atlas, chart_ambient_indices, chart_centers_indices = best_artifacts

    # 3. SERIALIZE ARTIFACTS
    torch.save(data_ambient.cpu(), os.path.join(args.output_dir, "data.pt"))
    torch.save(membership_mask, os.path.join(args.output_dir, "membership_mask.pt"))
    torch.save(chart_intrinsic_coords, os.path.join(args.output_dir, "chart_intrinsic_coords.pt"))
    torch.save(whitney_atlas, os.path.join(args.output_dir, "whitney_atlas.pt"))
    torch.save(chart_ambient_indices, os.path.join(args.output_dir, "chart_ambient_indices.pt"))
    torch.save(chart_centers_indices, os.path.join(args.output_dir, "chart_centers_indices.pt"))
    
    # Dump optimal parameter state matrix
    with open(os.path.join(args.output_dir, "best_empirical_params.yaml"), 'w') as f:
        yaml.dump(best_params, f)
        
    print(f"Phase 1 Grid Search Complete -> Radius-Driven Artifacts and Dynamic Regularity successfully serialized. Best Objective Score: {best_score}")

if __name__ == "__main__": 
    main()