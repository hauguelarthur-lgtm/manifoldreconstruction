import torch
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.manifoldclustering import construct_whitney_atlas

def main():
    default_data_path = os.path.join(project_root, "data", "raw", "dataset.pt")
    default_output_dir = os.path.join(project_root, "data", "processed")
    default_config_path = os.path.join(project_root, "configs", "default_config.yaml")

    parser = argparse.ArgumentParser(description="Constructs the intrinsic Whitney Submanifold Atlas.")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--config", type=str, default=default_config_path)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    d = config['manifold']['intrinsic_dim']
    num_charts= config['geometry']['num_charts']
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load ambient raw dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Fatal: Raw ambient dataset not found at {args.data_path}. "
                                f"Execute scripts/00_generate_basic_manifold.py first.")
    
    data_ambient = torch.load(args.data_path, map_location=device)
    
    # 2. Extract intrinsic dimension 'd' from config


    print(f"Loaded ambient tensor: Shape {data_ambient.shape} on {device}")
    print(f"Executing Intrinsic Whitney Atlas Construction: m={num_charts}, d={d}...")

    # 3. Execute Whitney Submanifold Decomposition
# In scripts/01_cluster_data.py, update the execution and saving block:

    (membership_mask, 
     chart_intrinsic_coords, 
     whitney_atlas, 
     cluster_centers, 
     smooth_sigmas,
     chart_ambient_indices) = construct_whitney_atlas(
        data=data_ambient,
        num_charts=num_charts,
        intrinsic_dim=d
    )

    torch.save(data_ambient.cpu(), os.path.join(args.output_dir, "data.pt"))
    torch.save(membership_mask, os.path.join(args.output_dir, "membership_mask.pt"))
    torch.save(chart_intrinsic_coords, os.path.join(args.output_dir, "chart_intrinsic_coords.pt"))
    torch.save(whitney_atlas, os.path.join(args.output_dir, "whitney_atlas.pt"))
    torch.save(cluster_centers, os.path.join(args.output_dir, "cluster_centers.pt"))
    torch.save(smooth_sigmas.cpu(), os.path.join(args.output_dir, "smooth_sigmas.pt"))
    torch.save(chart_ambient_indices, os.path.join(args.output_dir, "chart_ambient_indices.pt"))

    print(f"\nPhase 1 Complete. Serialized artifacts to {args.output_dir}:")
    print(f" ├── data.pt                    (Ambient Ground Truth, {data_ambient.shape})")
    print(f" ├── membership_mask.pt         (Overlapping chart assignments, {membership_mask.shape})")
    print(f" ├── chart_intrinsic_coords.pt  (List of {len(chart_intrinsic_coords)} intrinsic tensors in R^{d})")
    print(f" ├── whitney_atlas.pt           (List of {len(whitney_atlas)} Whitney tangent frames [mu_i, Q_i, W_i])")
    print(f" ├── cluster_centers.pt         (Ambient chart centroids in R^{data_ambient.shape[1]})")
    print(f" ├── smooth_sigmas.pt           (Compact bump function length-scales)")
    print(f" └── chart_ambient_indices.pt   (List of ambient index mappings per chart)")

if __name__ == "__main__":
    main()