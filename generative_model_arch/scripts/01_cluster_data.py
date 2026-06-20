# Partitions ambient data into topological patches.
import torch
import os
import sys
import argparse

# Robust path resolution and module import access
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.manifoldclustering import partition_data

def main():
    default_data_path = os.path.join(project_root, "data", "raw", "dataset.pt")
    default_output_dir = os.path.join(project_root, "data", "processed")

    parser = argparse.ArgumentParser(description="Partitions ambient data into topological patches.")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to raw empirical data tensor (N, p).")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save clustered data.")
    parser.add_argument("--num_charts", type=int, default=10, help="Number of local Euclidean charts (m).")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw empirical data: A ~ \mu^*
    # Fallback to random data for execution demonstration if file does not exist
    if os.path.exists(args.data_path):
        data = torch.load(args.data_path)
    else:
        print(f"Warning: {args.data_path} not found. Generating mock empirical data.")
        data = torch.randn(5000, 16) # N=5000, p=16

    print(f"Partitioning data into {args.num_charts} topological charts...")
    labels, cluster_centers, cluster_precisions = partition_data(data, num_charts=args.num_charts)

    torch.save(data, os.path.join(args.output_dir, "data.pt"))
    torch.save(labels, os.path.join(args.output_dir, "labels.pt"))
    torch.save(cluster_centers, os.path.join(args.output_dir, "cluster_centers.pt"))
    torch.save(cluster_precisions, os.path.join(args.output_dir, "cluster_precisions.pt"))
    print(f"Clustering complete. Artifacts saved to {args.output_dir}.")

if __name__ == "__main__":
    main()