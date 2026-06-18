# Partitions ambient data into topological patches.
import torch
import os
import argparse
from src.solver.clustering import partition_data

def main():
    parser = argparse.ArgumentParser(description="Partitions ambient data into topological patches.")
    parser.add_argument("--data_path", type=str, default="../data/raw/dataset.pt", help="Path to raw empirical data tensor (N, p).")
    parser.add_argument("--output_dir", type=str, default="../data/processed/", help="Directory to save clustered data.")
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
    labels, cluster_centers = partition_data(data, num_charts=args.num_charts)

    torch.save(data, os.path.join(args.output_dir, "data.pt"))
    torch.save(labels, os.path.join(args.output_dir, "labels.pt"))
    torch.save(cluster_centers, os.path.join(args.output_dir, "cluster_centers.pt"))
    print(f"Clustering complete. Artifacts saved to {args.output_dir}.")

if __name__ == "__main__":
    main()