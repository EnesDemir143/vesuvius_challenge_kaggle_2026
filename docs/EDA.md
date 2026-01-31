# Exploratory Data Analysis (EDA)

Vesuvius Surface Detection dataset analysis.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total volumes in CSV | 806 |
| Valid volumes (with images) | 786 |
| Missing images (deprecated) | 20 |
| Missing labels | 0 |
| Volume shape | (320, 320, 320) |
| Volume dtype | uint8 |
| Total voxels per volume | 32,768,000 |
| Volume size | ~32.8 MB |

## Label Encoding

| Value | Class | Description |
|-------|-------|-------------|
| 0 | Background | Non-papyrus regions |
| 1 | Foreground | Papyrus layer surface |
| 2 | Unlabeled | Not annotated regions |

## Scroll Distribution

| Scroll ID | Volume Count | Percentage |
|-----------|--------------|------------|
| 34117 | 376 | 47.8% |
| 35360 | 170 | 21.6% |
| 26010 | 129 | 16.4% |
| 26002 | 82 | 10.4% |
| 44430 | 16 | 2.0% |
| 53997 | 13 | 1.7% |

**Total scrolls:** 6

## Train/Val Split

Scroll-based stratified split (no data leakage):

| Split | Volumes | Scrolls |
|-------|---------|---------|
| Train | 616 | 26002, 26010, 34117, 44430, 53997 |
| Val | 170 | 35360 |

**Val ratio:** ~21.6%

## Class Distribution (Sample)

Typical label distribution in a volume:

| Class | Voxels | Percentage |
|-------|--------|------------|
| Background (0) | ~12.2M | ~37% |
| Foreground (1) | ~1.6M | ~5% |
| Unlabeled (2) | ~18.9M | ~58% |

> **Note:** Most of the volume is unlabeled. Use `ignore_index=2` in loss function.

## Key Observations

1. **Severe class imbalance** - Foreground is only ~5% of labeled data
2. **Sparse annotations** - ~58% of voxels are unlabeled
3. **Consistent resolution** - All volumes are 320³
4. **Scroll imbalance** - Scroll 34117 has 48% of all data
