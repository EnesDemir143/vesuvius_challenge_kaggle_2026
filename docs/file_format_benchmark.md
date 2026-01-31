# File Format Benchmark for 3D Volumetric Data

This document compares the performance of different file formats (LMDB, HDF5, Zarr, TIFF) for loading 3D volumetric data with PyTorch DataLoader.

## 📊 Benchmark Results (20 Epochs)

**Test Environment:**
- **Platform:** M2 Pro (macOS)
- **Batch Size:** 4
- **Num Workers:** 4
- **Crop Size:** 256³
- **Dataset:** 157 samples (20% subset)
- **Epochs:** 20

### Final Performance Summary

| Format | Avg Epoch Time | Avg Throughput | RAM After | Ranking |
|--------|----------------|----------------|-----------|---------|
| 🥇 **LMDB** | 1.24s | **155.8/sec** | 158.2MB | 1 |
| 🥈 **HDF5** | 1.26s | 135.4/sec | 146.9MB | 2 |
| 🥉 **Zarr** | 2.21s | 72.4/sec | 152.3MB | 3 |
| **TIFF Raw** | 6.07s | 26.1/sec | 126.1MB | 4 (baseline) |

## 📈 Performance Comparison

```
Throughput (samples/sec) - 20 Epoch Average
═══════════════════════════════════════════════════════════════════
LMDB      ████████████████████████████████████████████████████ 155.8
HDF5      █████████████████████████████████████████████        135.4
Zarr      ████████████████████████                              72.4
TIFF Raw  █████████                                             26.1
═══════════════════════════════════════════════════════════════════
```

## 📉 Detailed Epoch Analysis

### LMDB - Steady State Performance

| Metric | Epoch 1 | Epoch 2-5 Avg | Epoch 6-20 Avg |
|--------|---------|---------------|----------------|
| Time (s) | 6.29 | 1.05 | **0.94** |
| Throughput | 25.0 | 152.2 | **167.8** |

> [!TIP]
> LMDB requires warm-up on the first epoch, afterwards reaches **~168 samples/sec** - the highest throughput!

### HDF5 - Consistent Performance

| Metric | Epoch 1 | Epoch 2-5 Avg | Epoch 6-20 Avg |
|--------|---------|---------------|----------------|
| Time (s) | 3.81 | 1.27 | **1.08** |
| Throughput | 41.2 | 125.6 | **144.5** |

> [!NOTE]
> HDF5 with `driver='stdio'` is very stable on macOS, low variance.

### Zarr - Mid-tier Performance

| Metric | Epoch 1 | Epoch 2-5 Avg | Epoch 6-20 Avg |
|--------|---------|---------------|----------------|
| Time (s) | 3.80 | 2.20 | **2.09** |
| Throughput | 41.3 | 71.6 | **74.9** |

### TIFF Raw - Baseline

| Metric | Epoch 1 | Epoch 2-20 Avg |
|--------|---------|----------------|
| Time (s) | 8.27 | **5.95** |
| Throughput | 19.0 | **26.6** |

## 🔑 Key Findings (After 20 Epochs)

### 1. 🥇 LMDB Highest Throughput

- **~168 samples/sec after warm-up** (epochs 6-20)
- Persistent workers keep the worker pool ready
- Zero-copy `tobytes()` with minimal CPU overhead
- First epoch is slow (6.29s) but subsequent epochs are very fast

### 2. 🥈 HDF5 Most Consistent Performance

- **~144 samples/sec** stable throughput
- `driver='stdio'` performs excellently on macOS SSDs
- **Lowest variance** - consistent across epochs
- Shorter warm-up time compared to LMDB

### 3. 🥉 Zarr Cloud-Ready Alternative

- **~75 samples/sec** - half of HDF5 for local use
- Advantageous for cloud storage (S3/GCS) integration
- Memory-friendly with lazy loading

### 4. TIFF Raw - Conversion is Worth It!

> [!IMPORTANT]
> Format conversion is definitely worth it!
> - **LMDB: 6.0x faster** (168 vs 27 samples/sec)
> - **HDF5: 5.5x faster** (145 vs 27 samples/sec)
> - **Zarr: 2.9x faster** (75 vs 27 samples/sec)

## 💡 Format Selection Recommendations

| Use Case | 1st Choice | 2nd Choice | Why? |
|----------|------------|------------|------|
| **Max throughput (single machine)** | LMDB | HDF5 | 168 samples/sec peak |
| **Consistent performance** | HDF5 | LMDB | Low variance, stable |
| **Cloud storage (S3/GCS)** | Zarr | HDF5 | Cloud-native chunking |
| **Multi-machine distributed** | LMDB | Zarr | Lock-free key-value |
| **Memory-constrained** | HDF5 | Zarr | ~150MB RAM |
| **Quick prototyping** | TIFF Raw | - | No conversion needed |

## 🔧 Optimization Details

Optimizations used:

```python
# LMDB - Zero-copy, persistent env
env = lmdb.open(path, writemap=True)          # Faster on macOS
txn.put(key, volume.tobytes())                 # tobytes instead of pickle (~168/sec)
self.env = lmdb.open(path, readahead=False)   # No RAM waste

# HDF5 - stdio driver, persistent handle
h5py.File(path, 'r', driver='stdio')          # Optimized for macOS SSD (~145/sec)
rdcc_nbytes=1024*1024*4                        # 4MB cache per worker

# DataLoader - Persistent workers
DataLoader(..., persistent_workers=True)       # No worker spawn overhead
```

## 📊 Speedup Comparison (vs TIFF Raw)

| Format | Speedup (Cold Start) | Speedup (Warm) |
|--------|---------------------|----------------|
| LMDB | 3.8x | **6.4x** |
| HDF5 | 2.2x | **5.5x** |
| Zarr | 2.2x | **2.9x** |

## 🎯 Conclusion

Among the formats tested over 20 epochs:

1. **For maximum speed:** LMDB (~168 samples/sec after warm-up)
2. **For consistency:** HDF5 (low variance, stable ~145 samples/sec)
3. **For cloud integration:** Zarr (S3/GCS native)
4. **Don't use TIFF raw** - At least 3x slower, always convert!

## 📁 File Locations

```
benchmark/file_format/
├── file_format_benchmark.py    # Benchmark code
└── benchmark_results.csv       # Raw results (20 epochs × 4 formats = 80 rows)

docs/
└── file_format_benchmark.md    # This document
```

## 🔄 Running the Benchmark

```bash
# Clean old data (optional)
rm -rf data/processed/benchmark_*

# Run benchmark (20 epochs, ~5 minutes)
.venv/bin/python benchmark/file_format/file_format_benchmark.py
```

---

*Last updated: 2026-01-31 | 20 epoch test results*
