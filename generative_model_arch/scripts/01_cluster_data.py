import torch
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.manifoldclustering import construct_whitney_atlas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(project_root, "data", "raw", "dataset.pt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    d = int(config['manifold']['intrinsic_dim'])
    
    raw_num_charts = config['geometry']['num_charts']
    num_charts_arg = 'auto' if str(raw_num_charts).strip().lower() in ['auto', 'none', '0'] else int(float(raw_num_charts))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_ambient = torch.load(args.data_path, map_location=device)
    
    # Unpack the 4 standard immutable artifacts (\beta permanently locked to 1.5)
    (membership_mask, chart_intrinsic_coords, whitney_atlas, chart_ambient_indices) = construct_whitney_atlas(
        data=data_ambient,
        intrinsic_dim=d,
        num_charts=num_charts_arg,
        target_beta=1.50,
        packing_multiplier=3.0
    )

    torch.save(data_ambient.cpu(), os.path.join(args.output_dir, "data.pt"))
    torch.save(membership_mask, os.path.join(args.output_dir, "membership_mask.pt"))
    torch.save(chart_intrinsic_coords, os.path.join(args.output_dir, "chart_intrinsic_coords.pt"))
    torch.save(whitney_atlas, os.path.join(args.output_dir, "whitney_atlas.pt"))
    torch.save(chart_ambient_indices, os.path.join(args.output_dir, "chart_ambient_indices.pt"))
    
    # Clean up legacy dynamic beta artifact if present
    beta_file = os.path.join(args.output_dir, "besov_beta.pt")
    if os.path.exists(beta_file): os.remove(beta_file)
    print("Phase 1 Complete -> Artifacts serialized (besov_beta permanently locked to physical constant 1.50)")

if __name__ == "__main__": main()