"""
Compute mean and std from train split and save to dataset/stats/mean_std.npy
Optimized for Apple M2 Pro with parallel DataLoader.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VesuviusLMDBDataset


def compute_mean_std(
    lmdb_path: str, 
    val_ratio: float = 0.2, 
    seed: int = 42,
    num_workers: int = 4
) -> dict:
    """
    Compute channel-wise mean and std from training data.
    Uses parallel DataLoader for fast loading on M2 Pro.
    
    Note: batch_size=1 because volumes have different shapes.
    Parallelism comes from prefetching with multiple workers.
    
    Args:
        lmdb_path: Path to LMDB database
        val_ratio: Validation ratio for split
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers for data loading
    
    Returns:
        Dictionary with 'mean' and 'std' numpy arrays
    """
    print(f"Loading train dataset (num_workers={num_workers})...")
    
    dataset = VesuviusLMDBDataset(
        lmdb_path=lmdb_path,
        transform=None,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
        return_metadata=False
    )
    
    print(f"Train dataset size: {len(dataset)} volumes")
    
    # Create DataLoader with multiprocessing
    # batch_size=1 because volumes have different shapes
    # Parallelism achieved through prefetching with workers
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Must be 1 since volumes have different shapes
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    # Two-pass approach with vectorized computation
    print("Computing mean (pass 1/2)...")
    total_sum = 0.0
    total_count = 0
    
    for batch in tqdm(dataloader, desc="Computing mean"):
        image = batch['image'].squeeze(0).double()  # Remove batch dim
        total_sum += image.sum().item()
        total_count += image.numel()
    
    mean = total_sum / total_count
    
    print("Computing std (pass 2/2)...")
    total_sq_diff = 0.0
    
    for batch in tqdm(dataloader, desc="Computing std"):
        image = batch['image'].squeeze(0).double()
        total_sq_diff += ((image - mean) ** 2).sum().item()
    
    variance = total_sq_diff / total_count
    std = np.sqrt(variance)
    
    print(f"\nComputed statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Total voxels processed: {total_count:,}")
    
    return {
        'mean': np.array([mean], dtype=np.float32),
        'std': np.array([std], dtype=np.float32)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute mean/std from train split")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of DataLoader workers (default: 4 for M2 Pro)")
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    lmdb_path = project_root / "dataset" / "processed" / "train.lmdb"
    stats_dir = project_root / "dataset" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    output_path = stats_dir / "mean_std.npy"
    
    # Check if LMDB exists
    if not lmdb_path.exists():
        print(f"Error: LMDB database not found at {lmdb_path}")
        sys.exit(1)
    
    # Compute mean and std with parallel data loading
    stats = compute_mean_std(
        lmdb_path=str(lmdb_path),
        val_ratio=0.2,
        seed=42,
        num_workers=args.num_workers
    )
    
    # Save to npy file
    np.save(output_path, stats)
    print(f"\nStatistics saved to: {output_path}")
    
    # Verify by loading
    loaded = np.load(output_path, allow_pickle=True).item()
    print(f"Verification - Loaded mean: {loaded['mean']}, std: {loaded['std']}")


if __name__ == "__main__":
    main()
