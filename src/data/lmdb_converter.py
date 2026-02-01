import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import zoom
from tqdm import tqdm


def parse_csv(csv_path: Path) -> pd.DataFrame:
    """Parse CSV file with volume metadata."""
    df = pd.read_csv(csv_path)
    # Ensure id is string for consistent key handling
    df['id'] = df['id'].astype(str)
    return df


def find_volume_files(
    volume_id: str,
    image_dir: Path,
    label_dir: Optional[Path] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find image and label files for a given volume ID."""
    image_path = image_dir / f"{volume_id}.tif"
    label_path = None
    
    if not image_path.exists():
        return None, None
    
    if label_dir is not None:
        label_path = label_dir / f"{volume_id}.tif"
        if not label_path.exists():
            label_path = None
    
    return image_path, label_path


def estimate_lmdb_size(
    df: pd.DataFrame,
    image_dir: Path,
    target_size: Optional[Tuple[int, int, int]] = None,
    sample_count: int = 5
) -> int:
    """Estimate LMDB map size based on target size or sample files."""
    if target_size is not None:
        # Calculate based on target size (much more accurate)
        target_bytes = target_size[0] * target_size[1] * target_size[2]  # uint8
        # Image + label + metadata overhead (~2.2x)
        per_volume = target_bytes * 2.2
        # Total + 30% overhead for LMDB metadata
        estimated_size = int(per_volume * len(df) * 1.3)
        return max(estimated_size, 100 * 1024 * 1024)  # At least 100MB
    
    # Fallback: sample-based estimation for no-resize mode
    sample_ids = df['id'].head(min(sample_count, len(df))).tolist()
    total_size = 0
    valid_samples = 0
    
    for vol_id in sample_ids:
        image_path = image_dir / f"{vol_id}.tif"
        if image_path.exists():
            volume = tifffile.imread(image_path)
            # Image + label (estimate same size) + metadata overhead
            total_size += volume.nbytes * 2.2
            valid_samples += 1
    
    if valid_samples == 0:
        return 50 * 1024 * 1024 * 1024  # 50GB default
    
    avg_size = total_size / valid_samples
    # Total estimate + 30% overhead for LMDB metadata and fragmentation
    estimated_size = int(avg_size * len(df) * 1.3)
    
    return max(estimated_size, 100 * 1024 * 1024)  # At least 100MB


def convert_to_lmdb(
    csv_path: Path,
    image_dir: Path,
    label_dir: Optional[Path],
    output_path: Path,
    overwrite: bool = False,
    max_samples: Optional[int] = None,
    target_size: Optional[Tuple[int, int, int]] = (256, 256, 256)
) -> Dict:
    # Handle existing LMDB
    if output_path.exists():
        if not overwrite:
            print(f"LMDB already exists: {output_path}")
            print("Use --overwrite to replace it.")
            return {"status": "skipped"}
        
        import shutil
        shutil.rmtree(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse CSV
    df = parse_csv(csv_path)
    print(f"Found {len(df)} volumes in CSV")
    
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limiting to {max_samples} samples for testing")
    
    # Estimate map size
    map_size = estimate_lmdb_size(df, image_dir, target_size=target_size)
    print(f"Estimated LMDB size: {map_size / (1024**3):.2f} GB")
    
    start_time = time.time()
    stats = {
        "total": len(df),
        "converted": 0,
        "with_labels": 0,
        "missing_images": 0,
        "scroll_ids": {}
    }
    
    # Open LMDB environment
    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        writemap=True
    )
    
    volume_keys = []
    
    with env.begin(write=True) as txn:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            vol_id = str(row['id'])
            scroll_id = int(row['scroll_id'])
            
            # Find files
            image_path, label_path = find_volume_files(
                vol_id, image_dir, label_dir
            )
            
            if image_path is None:
                stats["missing_images"] += 1
                print(f"Warning: Image not found for volume {vol_id}")
                continue
            
            # Load image volume
            image = tifffile.imread(image_path)
            
            # Resize to target size if specified
            if target_size is not None and image.shape != target_size:
                zoom_factors = tuple(t / s for t, s in zip(target_size, image.shape))
                image = zoom(image, zoom_factors, order=1)  # Trilinear interpolation
                image = image.astype(np.uint8)  # Ensure uint8 after interpolation
            
            # Store image data
            txn.put(f"{vol_id}_image".encode(), image.tobytes())
            txn.put(
                f"{vol_id}_image_shape".encode(),
                np.array(image.shape, dtype=np.int64).tobytes()
            )
            txn.put(f"{vol_id}_image_dtype".encode(), str(image.dtype).encode())
            
            # Store metadata
            txn.put(f"{vol_id}_scroll_id".encode(), str(scroll_id).encode())
            
            # Load and store label if available
            has_label = label_path is not None
            txn.put(f"{vol_id}_has_label".encode(), str(has_label).encode())
            
            if has_label:
                label = tifffile.imread(label_path)
                
                # Resize label to target size (use nearest neighbor to preserve discrete values)
                if target_size is not None and label.shape != target_size:
                    zoom_factors = tuple(t / s for t, s in zip(target_size, label.shape))
                    label = zoom(label, zoom_factors, order=0)  # Nearest neighbor
                
                txn.put(f"{vol_id}_label".encode(), label.tobytes())
                txn.put(
                    f"{vol_id}_label_shape".encode(),
                    np.array(label.shape, dtype=np.int64).tobytes()
                )
                txn.put(f"{vol_id}_label_dtype".encode(), str(label.dtype).encode())
                stats["with_labels"] += 1
            
            volume_keys.append(vol_id)
            stats["converted"] += 1
            
            # Track scroll_id distribution
            if scroll_id not in stats["scroll_ids"]:
                stats["scroll_ids"][scroll_id] = 0
            stats["scroll_ids"][scroll_id] += 1
        
        # Store index of all volume IDs
        txn.put(b"__keys__", json.dumps(volume_keys).encode())
        txn.put(b"__len__", str(len(volume_keys)).encode())
    
    env.close()
    
    conversion_time = time.time() - start_time
    stats["conversion_time_seconds"] = conversion_time
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Converted: {stats['converted']}/{stats['total']} volumes")
    print(f"With labels: {stats['with_labels']}")
    print(f"Missing images: {stats['missing_images']}")
    print(f"Time: {conversion_time:.2f}s")
    print(f"\nScroll ID distribution:")
    for sid, count in sorted(stats["scroll_ids"].items()):
        print(f"  {sid}: {count} volumes")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIFF files to LMDB format for fast data loading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="Path to CSV file with volume metadata (id, scroll_id)"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing input TIFF images. If not specified, inferred from CSV path."
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="Directory containing input TIFF labels. If not specified, inferred from CSV path."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Output directory for LMDB database"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing LMDB database"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (for testing)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip label conversion (for test set)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        metavar=("D", "H", "W"),
        help="Target volume size (D H W). Set to 0 0 0 to disable resizing."
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / args.csv_file if not args.csv_file.is_absolute() else args.csv_file
    
    # Infer image/label directories from CSV path if not specified
    csv_dir = csv_path.parent
    
    if args.image_dir is None:
        # Check if this is train or test based on CSV filename
        if "train" in csv_path.stem:
            args.image_dir = csv_dir / "train_images"
        elif "test" in csv_path.stem:
            args.image_dir = csv_dir / "test_images"
        else:
            raise ValueError("Could not infer image directory. Please specify --image-dir")
    
    image_dir = project_root / args.image_dir if not args.image_dir.is_absolute() else args.image_dir
    
    label_dir = None
    if not args.no_labels:
        if args.label_dir is None:
            if "train" in csv_path.stem:
                args.label_dir = csv_dir / "train_labels"
        
        if args.label_dir is not None:
            label_dir = project_root / args.label_dir if not args.label_dir.is_absolute() else args.label_dir
            if not label_dir.exists():
                print(f"Warning: Label directory not found: {label_dir}")
                label_dir = None
    
    output_dir = project_root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    
    # Determine output filename
    if "train" in csv_path.stem:
        output_path = output_dir / "train.lmdb"
    elif "test" in csv_path.stem:
        output_path = output_dir / "test.lmdb"
    else:
        output_path = output_dir / "dataset.lmdb"
    
    print("=" * 60)
    print("VESUVIUS LMDB CONVERTER")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Output: {output_path}")
    print()
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    # Parse target size
    target_size = tuple(args.target_size) if any(s > 0 for s in args.target_size) else None
    if target_size:
        print(f"Target size: {target_size}")
    else:
        print("Target size: None (no resizing)")
    
    convert_to_lmdb(
        csv_path=csv_path,
        image_dir=image_dir,
        label_dir=label_dir,
        output_path=output_path,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        target_size=target_size
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
