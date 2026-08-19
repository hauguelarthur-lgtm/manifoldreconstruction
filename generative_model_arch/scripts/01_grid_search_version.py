import torch
import os
import sys
import argparse
import yaml
import itertools
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Modular Imports from the new architecture
from src.manifoldclustering import construct_whitney_atlas, EmpiricalConfig
from src.algebraic_engine import apply_algebraic_taylor_regression, AlgebraicWhitneyEvaluator
from src.tests_metric import test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(project_root, "data", "raw", "dataset.pt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config_yaml = yaml.safe_load(f)
        
    d = int(config_yaml['manifold']['intrinsic_dim'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data from {args.data_path}")
    data_ambient = torch.load(args.data_path).to(device)
    N = data_ambient.size(0)

    # 2. Pre-Search Pruning
    # Calculate ceiling dynamically based on dataset size and dimension
    print(f"\n--- Pre-Search Analysis ---")
    print(f"Dataset Size (N): {N}, Intrinsic Dim (d): {d}")


    # Lambda parameters removed; they are handled empirically inside algebraic_engine.py
    # C_overlap 1.0 is removed in favor of Safe Harbor blending [1.4 - 1.8]
    grid_space = {
        'volume_scale': [5, 6,7,8,10,15,20],
        'C_overlap': [1.4], 
        'beta': [0.5,1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    }

    keys, values = zip(*grid_space.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = float('inf')
    best_params = None
    best_artifacts = None

    print(f"Starting Grid Search over {len(permutations)} valid configurations...\n")

    # 3. Main Grid Search Loop
    for idx, params in enumerate(permutations):
        
            # A. Configure Empirical Constants
        empirical_config = EmpiricalConfig(
                volume_scale=params['volume_scale'],
                C_overlap=params['C_overlap'],
                beta=params['beta']
            )
            
            # B. Topological Hub (1st-Order Affine Extraction)
        (global_membership_mask, 
             atlas_frames,
             intrinsic_coords, 
             chart_ambient_indices, 
             fps_centers) = construct_whitney_atlas(
                 data_ambient, 
                 intrinsic_dim=d, 
                 empirical_config=empirical_config
             )
             
            # C. Algebraic Engine (Higher-Order Taylor Tensors)
        augmented_atlas = apply_algebraic_taylor_regression(
                data_ambient=data_ambient,
                chart_ambient_indices=chart_ambient_indices,
                atlas_frames=atlas_frames,
                d=d,
                beta=params['beta'],
                device=device
            )
            
            # D. Algebraic Evaluator Instantiation
        k_degree = math.floor(params['beta']) + 1
        algebraic_manifold = AlgebraicWhitneyEvaluator(
                augmented_atlas=augmented_atlas, 
                k_degree=k_degree, 
                device=device
            )
            
            # E. Upgraded Structural Risk Evaluation
        score = test_metrics(
                global_manifold=algebraic_manifold,
                fps_centers=fps_centers,
                global_membership_mask=global_membership_mask,
                data_ambient=data_ambient,
                d=d,
                beta=params['beta']
            )
        print(score)
            # Log and track best geometric configuration
        if score < best_score:
            best_score = score
            best_params = params
            best_artifacts = (global_membership_mask, intrinsic_coords, augmented_atlas, chart_ambient_indices, fps_centers)
            print(f"[{idx+1}/{len(permutations)}] New best score: {best_score:.6f} via {best_params}")
        
        else:
            del global_membership_mask
            del algebraic_manifold
            del augmented_atlas
            torch.cuda.empty_cache()


    if best_score == float('inf') or best_artifacts is None:
        raise RuntimeError("Grid search failed to resolve any mathematically viable configurations.")

    print(f"\nGrid Search Complete. Best Score: {best_score:.6f}")

    # 4. Unpack and Serialize Optimal Artifacts
    membership_mask, chart_intrinsic_coords, optimal_augmented_atlas, chart_ambient_indices, fps_centers = best_artifacts

    torch.save(data_ambient.cpu(), os.path.join(args.output_dir, "data.pt"))
    torch.save(membership_mask, os.path.join(args.output_dir, "membership_mask.pt"))
    torch.save(chart_intrinsic_coords, os.path.join(args.output_dir, "chart_intrinsic_coords.pt"))
    torch.save(optimal_augmented_atlas, os.path.join(args.output_dir, "whitney_atlas.pt"))
    torch.save(chart_ambient_indices, os.path.join(args.output_dir, "chart_ambient_indices.pt"))
    torch.save(fps_centers.cpu(), os.path.join(args.output_dir, "fps_centers.pt"))
    
    # Dump optimal parameter state matrix
    with open(os.path.join(args.output_dir, "best_empirical_params.yaml"), 'w') as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    main()