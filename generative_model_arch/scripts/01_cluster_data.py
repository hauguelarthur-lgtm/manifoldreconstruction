import torch
import os
import sys
import argparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.manifoldclustering import construct_whitney_atlas, EmpiricalConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(project_root, "data", "raw", "dataset.pt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "processed"))
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config = yaml.safe_load(f)
        
    d = int(config['manifold']['intrinsic_dim'])
    

    # 2. INITIALIZE EMPIRICAL TUNING CONFIGURATION
    # Maps real-world dataset adjustments to the mathematical backend
    emp_cfg_dict = config.get('empirical_tuning', {})
    empirical_params = EmpiricalConfig(
        volume_scale=float(emp_cfg_dict.get('volume_scale', 2.0)),
        oversample_ratio=float(emp_cfg_dict.get('oversample_ratio', 1.5)),
        lambda_base=float(emp_cfg_dict.get('lambda_base', 1e-7)),
        lambda_trace_scale=float(emp_cfg_dict.get('lambda_trace_scale', 1e-4)),
        max_radius_cap=emp_cfg_dict.get('max_radius_cap', None),
        beta=emp_cfg_dict.get('beta',1.5)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_ambient = torch.load(args.data_path, map_location=device)
    
    # 3. EXECUTE RADIUS-DRIVEN ATLAS CONSTRUCTION
    # Removed legacy 'num_charts' and 'packing_multiplier' arguments
    print(f"Executing Whitney Partition... Target Beta: {beta}")
    (membership_mask, chart_intrinsic_coords, whitney_atlas, chart_ambient_indices) = construct_whitney_atlas(
        data=data_ambient,
        intrinsic_dim=d,
        target_beta=beta,
        empirical_config=empirical_params
    )

    # 4. SERIALIZE ARTIFACTS
    torch.save(data_ambient.cpu(), os.path.join(args.output_dir, "data.pt"))
    torch.save(membership_mask, os.path.join(args.output_dir, "membership_mask.pt"))
    torch.save(chart_intrinsic_coords, os.path.join(args.output_dir, "chart_intrinsic_coords.pt"))
    torch.save(whitney_atlas, os.path.join(args.output_dir, "whitney_atlas.pt"))
    torch.save(chart_ambient_indices, os.path.join(args.output_dir, "chart_ambient_indices.pt"))
    
    # Save the dynamically resolved beta to ensure the SDE drift 
    # executes the exact corresponding polynomial combinations
    torch.save(torch.tensor(beta), os.path.join(args.output_dir, "besov_beta.pt"))
    
    print("Phase 1 Complete -> Radius-Driven Artifacts and Dynamic Regularity successfully serialized.")

if __name__ == "__main__": 
    main()