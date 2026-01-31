"""
File Format Benchmark for 3D Volumetric Data

Compares HDF5, LMDB, Zarr, and NPY formats for PyTorch DataLoader performance.
Optimized for M2 Pro with realistic benchmark configurations.
"""

import csv
import gc
import os
import random
import time
from pathlib import Path
from typing import List

import h5py
import lmdb
import numpy as np
import psutil
import tifffile
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Configuration
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "train_images"
PROCESSED_DIR = DATA_ROOT / "processed"
BENCHMARK_RESULTS_PATH = Path(__file__).parent / "benchmark_results.csv"

SUBSET_RATIO = 0.20  # Use 20% of data
CROP_SIZE = 256  # Center crop to 256x256x256
BATCH_SIZE = 4   # 256³ için 4-8 arası ideal, 16GB RAM'de 8 batch × 256³ ≈ 512MB
NUM_WORKERS = 4  # M2 Pro'da 4-6 ideal, 8 fazla contention yaratır
NUM_EPOCHS = 20

SEED = 42


def get_tiff_files(subset_ratio: float = SUBSET_RATIO) -> List[Path]:
    """Get a subset of TIFF files for benchmarking."""
    all_files = sorted(RAW_DIR.glob("*.tif"))
    random.seed(SEED)
    num_files = int(len(all_files) * subset_ratio)
    selected_files = random.sample(all_files, num_files)
    return selected_files


def center_crop_3d(volume: np.ndarray, crop_size: int) -> np.ndarray:
    """Center crop a 3D volume to the specified size."""
    d, h, w = volume.shape
    start_d = (d - crop_size) // 2
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return volume[
        start_d : start_d + crop_size,
        start_h : start_h + crop_size,
        start_w : start_w + crop_size,
    ]


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


# =============================================================================
# CONVERTERS
# =============================================================================


def convert_to_lmdb(tiff_files: List[Path], output_dir: Path) -> float:
    """Convert TIFF files to LMDB format (Zero-copy ready)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "data.lmdb"

    if db_path.exists():
        print(f"  LMDB already exists: {db_path}")
        return 0.0

    start_time = time.time()
    map_size = len(tiff_files) * CROP_SIZE**3 * 2

    env = lmdb.open(str(db_path), map_size=map_size, writemap=True)  # Faster on macOS

    with env.begin(write=True) as txn:
        for i, tiff_path in enumerate(tqdm(tiff_files, desc="  LMDB")):
            volume = tifffile.imread(tiff_path)
            volume = center_crop_3d(volume, CROP_SIZE)
            # ❌ Pickle kaldırıldı, direkt binary
            txn.put(f"{i}".encode(), volume.tobytes())
        txn.put(b"__len__", str(len(tiff_files)).encode())

    env.close()
    return time.time() - start_time


def convert_to_hdf5(tiff_files: List[Path], output_dir: Path, compressed: bool = False) -> float:
    """Convert TIFF files to HDF5 (Uncompressed for fair benchmark)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.h5"

    if output_path.exists():
        print(f"  HDF5 already exists: {output_path}")
        return 0.0

    start_time = time.time()

    # M2 Pro için optimize: Sıkıştırma yok (hız için), chunk'lar büyük
    with h5py.File(output_path, "w", libver='latest') as f:
        for i, tiff_path in enumerate(tqdm(tiff_files, desc="  HDF5")):
            volume = tifffile.imread(tiff_path)
            volume = center_crop_3d(volume, CROP_SIZE)

            kwargs = {
                "chunks": (64, 64, 64),  # 256/4 = 4 chunk per dim
                "compression": None  # ❌ Sıkıştırma yok (hız için)
            }
            if compressed:
                kwargs["compression"] = "gzip"

            f.create_dataset(f"volume_{i}", data=volume, **kwargs)

    return time.time() - start_time


def convert_to_zarr(tiff_files: List[Path], output_dir: Path) -> float:
    """Convert TIFF files to Zarr format. Returns conversion time in seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "data.zarr"

    if zarr_path.exists():
        print(f"  Zarr already exists: {zarr_path}")
        return 0.0

    start_time = time.time()

    root = zarr.open_group(str(zarr_path), mode="w")

    for i, tiff_path in enumerate(tqdm(tiff_files, desc="  Converting to Zarr")):
        volume = tifffile.imread(tiff_path)
        volume = center_crop_3d(volume, CROP_SIZE)
        root.create_array(
            f"volume_{i}",
            data=volume,
            chunks=(64, 64, 64),
        )

    return time.time() - start_time


def convert_to_npy(tiff_files: List[Path], output_dir: Path) -> float:
    """Convert TIFF files to NPY format. Returns conversion time in seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    if (output_dir / "volume_0.npy").exists():
        print(f"  NPY already exists: {output_dir}")
        return 0.0

    start_time = time.time()

    for i, tiff_path in enumerate(tqdm(tiff_files, desc="  Converting to NPY")):
        volume = tifffile.imread(tiff_path)
        volume = center_crop_3d(volume, CROP_SIZE)
        np.save(output_dir / f"volume_{i}.npy", volume)

    return time.time() - start_time


# =============================================================================
# PYTORCH DATASETS
# =============================================================================


class LMDBDataset(Dataset):
    """Optimized LMDB - Persistent env per worker."""

    def __init__(self, lmdb_path: Path):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        # Master process'te length al
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self._length = int(txn.get(b"__len__").decode())
        env.close()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Lazy init: Her worker'da bir kere açılır
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,        # Multi-process güvenli
                readahead=False,   # ❌ RAM israfı yok
                map_size=2**32
            )
        with self.env.begin() as txn:
            data = txn.get(f"{idx}".encode())
            # ❌ Pickle yok, zero-copy from buffer
            vol = np.frombuffer(data, dtype=np.uint8).reshape(CROP_SIZE, CROP_SIZE, CROP_SIZE)
            # .copy() eklemek zorundayız yoksa tensor buffer collision olur
            return torch.from_numpy(vol.copy()).float().unsqueeze(0)

    def close(self):
        if self.env:
            self.env.close()
            self.env = None


class MMapNPYDataset(Dataset):
    """Memory-mapped NPY - En hızlı ve RAM dostu."""

    def __init__(self, npy_dir: Path):
        self.npy_dir = npy_dir
        self.files = sorted(npy_dir.glob("volume_*.npy"))
        # ❌ Eager loading yok, sadece mmap
        self.mmaps = [np.load(f, mmap_mode='r') for f in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # mmap'den oku (sadece gerekli sayfalar cache'e gelir)
        return torch.from_numpy(self.mmaps[idx].copy()).float().unsqueeze(0)

    def close(self):
        # Mmap'leri temizle
        self.mmaps = None
        gc.collect()


class HDF5Dataset(Dataset):
    """HDF5 with driver='stdio' for M2 SSD speed."""

    def __init__(self, hdf5_path: Path):
        self.hdf5_path = str(hdf5_path)
        self.f = None
        # Ana process'te keys al ama kapat
        with h5py.File(self.hdf5_path, 'r', driver='stdio') as f:  # ❌ stdio driver (faster on macOS)
            self.keys = [k for k in f.keys() if k.startswith("volume_")]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Persistent file handle per worker
        if self.f is None:
            # ❌ Küçük cache (multi-worker'da RAM patlamasın)
            self.f = h5py.File(
                self.hdf5_path,
                'r',
                driver='stdio',
                rdcc_nbytes=1024*1024*4,  # Sadece 4MB cache per worker
                rdcc_w0=0.5
            )
        volume = self.f[self.keys[idx]][:]
        return torch.from_numpy(volume).float().unsqueeze(0)

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


class ZarrDataset(Dataset):
    """Zarr with lazy loading."""

    def __init__(self, zarr_path: Path):
        self.zarr_path = str(zarr_path)
        self.root = None
        # Master'da keys al
        root = zarr.open_group(self.zarr_path, mode="r")
        self.keys = [k for k in root.keys() if k.startswith("volume_")]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.root is None:
            self.root = zarr.open_group(self.zarr_path, mode="r")
        # ❌ Lazy indexing ([:] yerine direkt access lazy olur zarr'da)
        volume = self.root[self.keys[idx]]
        # Zarr zaten lazy, ama tensor'a çevirirken copy gerekli
        return torch.from_numpy(np.array(volume)).float().unsqueeze(0)

    def close(self):
        self.root = None


class TIFFDataset(Dataset):
    """Raw TIFF Dataset - Baseline comparison."""

    def __init__(self, tiff_files: List[Path]):
        self.tiff_files = tiff_files

    def __len__(self) -> int:
        return len(self.tiff_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        volume = tifffile.imread(self.tiff_files[idx])
        volume = center_crop_3d(volume, CROP_SIZE)
        return torch.from_numpy(volume).float().unsqueeze(0)

    def close(self):
        pass


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def benchmark_dataloader(
    dataset: Dataset,
    num_workers: int,
    num_epochs: int = NUM_EPOCHS,
) -> List[dict]:
    """Benchmark with persistent workers (realistic training)."""
    results = []

    # ❌ Persistent workers (epoch'lar arası worker spawn yok)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,  # ⚡ Kritik
        pin_memory=False,  # M2 Pro'da CPU-GPU unified memory yoksa False
        prefetch_factor=2 if num_workers > 0 else None
    )

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        # Daha doğru RAM ölçümü (sadece process değil, system context)
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        num_samples = 0

        # tqdm yavaşlatmasın diye sadece epoch başında
        print(f"    Epoch {epoch} starting...")
        for batch in loader:
            num_samples += batch.shape[0]
            # Simulate training step (0.001s GPU işlem simülasyonu)
            if torch.backends.mps.is_available():  # M2 Pro
                time.sleep(0.001)

        epoch_time = time.time() - start_time
        ram_after = process.memory_info().rss / (1024 * 1024)

        results.append({
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 3),
            "samples_per_sec": round(num_samples / epoch_time, 2),
            "ram_delta_mb": round(ram_after - ram_before, 2),
            "ram_after_mb": round(ram_after, 2),
            "num_samples": num_samples,
        })

        print(f"      Epoch {epoch} done: {epoch_time:.2f}s")

    return results


def run_benchmarks():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("FILE FORMAT BENCHMARK (Optimized for M2 Pro)")
    print("=" * 60)

    # Get subset of files
    tiff_files = get_tiff_files()
    print(f"\nUsing {len(tiff_files)} files ({SUBSET_RATIO*100:.0f}% of dataset)")
    print(f"Config: BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, CROP_SIZE={CROP_SIZE}")

    # Define format configurations
    formats = {
        "lmdb": {
            "output_dir": PROCESSED_DIR / "benchmark_lmdb",
            "converter": convert_to_lmdb,
            "dataset_class": LMDBDataset,
            "data_path": lambda d: d / "data.lmdb",
        },
        "hdf5": {
            "output_dir": PROCESSED_DIR / "benchmark_hdf5",
            "converter": convert_to_hdf5,
            "dataset_class": HDF5Dataset,
            "data_path": lambda d: d / "data.h5",
        },
        "zarr": {
            "output_dir": PROCESSED_DIR / "benchmark_zarr",
            "converter": convert_to_zarr,
            "dataset_class": ZarrDataset,
            "data_path": lambda d: d / "data.zarr",
        },
        "tiff_raw": {
            "output_dir": None,  # No conversion needed
            "converter": None,
            "dataset_class": TIFFDataset,
            "data_path": lambda d: tiff_files,  # Use raw files directly
        },
    }

    all_results = []
    conversion_times = {}

    # Convert all formats first
    print("\n" + "=" * 60)
    print("CONVERTING DATA")
    print("=" * 60)

    for fmt_name, fmt_config in formats.items():
        print(f"\n[{fmt_name.upper()}]")
        if fmt_config["converter"] is None:
            print("  No conversion needed (raw format)")
            conversion_times[fmt_name] = 0.0
            continue
        conv_time = fmt_config["converter"](tiff_files, fmt_config["output_dir"])
        conversion_times[fmt_name] = conv_time
        if conv_time > 0:
            print(f"  Conversion time: {conv_time:.2f}s")

    # Benchmark each format
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)

    for fmt_name, fmt_config in formats.items():
        print(f"\n[{fmt_name.upper()}]")

        data_path = fmt_config["data_path"](fmt_config["output_dir"])
        dataset = fmt_config["dataset_class"](data_path)
        print(f"  Dataset size: {len(dataset)} samples")

        print(f"  Testing with num_workers={NUM_WORKERS}...")

        epoch_results = benchmark_dataloader(dataset, NUM_WORKERS)

        for result in epoch_results:
            all_results.append({
                "format": fmt_name,
                "num_workers": NUM_WORKERS,
                "conversion_time_sec": round(conversion_times[fmt_name], 2),
                **result,
            })

        avg_time = sum(r["epoch_time_sec"] for r in epoch_results) / len(epoch_results)
        avg_throughput = sum(r["samples_per_sec"] for r in epoch_results) / len(epoch_results)
        print(f"    Avg epoch time: {avg_time:.2f}s, Avg throughput: {avg_throughput:.1f} samples/sec")

        dataset.close()
        gc.collect()

    # Save results to CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    fieldnames = [
        "format", "num_workers", "epoch", "epoch_time_sec",
        "samples_per_sec", "ram_delta_mb", "ram_after_mb", "num_samples", "conversion_time_sec"
    ]

    with open(BENCHMARK_RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {BENCHMARK_RESULTS_PATH}")

    print("\n" + "=" * 60)
    print(f"SUMMARY (Average across {NUM_EPOCHS} epochs, num_workers={NUM_WORKERS})")
    print("=" * 60)

    summary_data = {}
    for result in all_results:
        fmt = result["format"]
        if fmt not in summary_data:
            summary_data[fmt] = {"times": [], "throughputs": [], "ram_delta": [], "ram_after": []}
        summary_data[fmt]["times"].append(result["epoch_time_sec"])
        summary_data[fmt]["throughputs"].append(result["samples_per_sec"])
        summary_data[fmt]["ram_delta"].append(result["ram_delta_mb"])
        summary_data[fmt]["ram_after"].append(result["ram_after_mb"])

    print(f"\n{'Format':<12} {'Epoch Time':>12} {'Throughput':>15} {'RAM Delta':>12} {'RAM After':>12}")
    print("-" * 65)
    for fmt, data in summary_data.items():
        avg_time = sum(data["times"]) / len(data["times"])
        avg_throughput = sum(data["throughputs"]) / len(data["throughputs"])
        avg_ram_delta = sum(data["ram_delta"]) / len(data["ram_delta"])
        avg_ram_after = sum(data["ram_after"]) / len(data["ram_after"])
        print(f"{fmt:<12} {avg_time:>10.2f}s {avg_throughput:>12.1f}/sec {avg_ram_delta:>10.1f}MB {avg_ram_after:>10.1f}MB")

    print("\nBenchmark complete!")
    print("\nExpected ranking: LMDB ≈ NPY mmap > HDF5 > Zarr")


if __name__ == "__main__":
    run_benchmarks()
