# 🏛️ Vesuvius Challenge - 3D Surface Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io)

Antik Herculaneum parşömenlerinde **3D yüzey segmentasyonu** için derin öğrenme pipeline'ı.

## 📋 Proje Özeti

Bu proje, Vesuvius Challenge yarışması için geliştirilmiş bir 3D volumetrik segmentasyon çözümüdür. Mikro-CT tarama verilerinden parşömen yüzeylerini tespit etmeyi amaçlar.

### Temel Özellikler

- 🔬 **3D Volumetrik Segmentasyon** - 256×256×256 voxel volumes
- 🏗️ **Çoklu Model Desteği** - UNet, SegResNet, SwinUNETR, SegFormer3D
- 🚀 **LMDB ile Hızlı Veri Yükleme** - Standart TIFF'e göre ~20x hızlı
- ⚡ **Mixed Precision Training (AMP)** - Daha hızlı eğitim, daha az VRAM
- 📊 **Yarışma Metrikleri** - Surface Dice, VOI Score, Topo Score

---

## 📁 Proje Yapısı

```
Vesuvius_Challenge_Surface_Detection/
├── 📄 train.py                 # Ana eğitim scripti
├── 📄 config.yaml              # Tüm konfigürasyonlar
├── 📂 src/                     # Kaynak kod
│   ├── 📂 data/                # Veri işleme
│   │   ├── dataset.py          # VesuviusLMDBDataset sınıfı
│   │   ├── transforms.py       # Augmentation pipeline
│   │   └── lmdb_converter.py   # TIFF → LMDB dönüştürücü
│   ├── 📂 training/            # Eğitim modülleri
│   │   ├── trainer.py          # Training loop, checkpoint
│   │   └── experiment_logger.py # Loglama, metrik takibi
│   ├── 📂 metrics/             # Değerlendirme metrikleri
│   │   ├── surface_dice.py     # Surface Dice Score
│   │   ├── voi_score.py        # Variation of Information
│   │   └── topo_score.py       # Topology Score
│   └── 📂 utils/               # Yardımcı fonksiyonlar
├── 📂 docs/                    # Dokümantasyon
├── 📂 notebook/                # Jupyter notebooks
├── 📂 benchmark/               # Performans testleri
└── 📂 runs/                    # Eğitim çıktıları
```

---

## 🔧 Kod Modülleri

### 1. Veri Pipeline (`src/data/`)

#### `dataset.py` - VesuviusLMDBDataset
PyTorch Dataset sınıfı. LMDB veritabanından 3D volume'ları yükler.

```python
# Özellikler:
- LMDB'den hızlı okuma (mmap ile zero-copy)
- Scroll-based stratified train/val split
- Multiprocessing DataLoader uyumlu
```

#### `transforms.py` - Augmentation Pipeline
3D volumetrik veri için augmentation'lar.

| Transform | Açıklama |
|-----------|----------|
| `CenterCropOrPad` | Volume'u 256³'e kırp veya padding ekle |
| `ZJitter` | Z-ekseni boyunca rastgele kaydırma |
| `BasicAugs` | Flip, Rotate90 |
| `PipMix` | CutMix benzeri 3D mixing |
| `Normalize` | Mean/std normalizasyon |

#### `lmdb_converter.py` - TIFF → LMDB
Raw TIFF verilerini LMDB formatına dönüştürür (~20x hızlı yükleme).

---

### 2. Model Seçenekleri

| Model | Tip | Açıklama |
|-------|-----|----------|
| **UNet** | CNN | Klasik encoder-decoder, 32-512 channels |
| **SegResNet** | CNN | ResNet tabanlı, MONAI |
| **SwinUNETR** | Transformer | Swin Transformer + UNet decoder |
| **SegFormer3D** | Transformer | MixVisionTransformer backbone |

---

### 3. Training (`src/training/`)

#### `trainer.py` - Trainer Sınıfı
Tam özellikli training loop:

- ✅ Mixed Precision (AMP)
- ✅ Gradient Accumulation
- ✅ Gradient Clipping
- ✅ Learning Rate Scheduling (Cosine, Step, Plateau)
- ✅ Early Stopping
- ✅ Checkpoint (best.pth, last.pth)
- ✅ Resume from checkpoint

#### Fine-Tuning Modları

| Mod | Açıklama |
|-----|----------|
| `linear_probe` | Sadece head eğitilir, encoder freeze |
| `middle` | Encoder'ın yarısı freeze, layer-wise LR decay |
| `full` | Tüm model eğitilir, layer-wise LR decay |

---

### 4. Metrikler (`src/metrics/`)

| Metrik | Dosya | Açıklama |
|--------|-------|----------|
| **Surface Dice** | `surface_dice.py` | Yüzey mesafesi tabanlı Dice, τ toleransı |
| **VOI Score** | `voi_score.py` | Variation of Information, cluster uyumu |
| **Topo Score** | `topo_score.py` | Topolojik doğruluk (Betti numaraları) |
| **Competition Score** | `competition_metrics.py` | Ağırlıklı ortalama |

```python
competition_score = 0.5 * surface_dice + 0.3 * voi_score + 0.2 * topo_score
```

---

## 📚 Dokümantasyon

| Dosya | Açıklama |
|-------|----------|
| [docs/EDA.md](docs/EDA.md) | Veri keşif analizi |
| [docs/augmentation_pipeline.md](docs/augmentation_pipeline.md) | Augmentation detayları |
| [docs/file_format_benchmark.md](docs/file_format_benchmark.md) | LMDB vs HDF5 vs Zarr karşılaştırması |

---

## 📓 Notebooks

| Notebook | Açıklama |
|----------|----------|
| `01_data_exploration.ipynb` | Veri yapısı analizi, istatistikler |
| `02_data_visualization.ipynb` | 3D volume görselleştirme |
| `03_augmentation_visualization.ipynb` | Augmentation efektleri |

---

## ⚙️ Konfigürasyon

`config.yaml` dosyasında tüm ayarlar:

```yaml
data:
  train_lmdb: "dataset/processed/train.lmdb"
  val_ratio: 0.2

training:
  model: "segresnet"       # unet, segresnet, swinunetr, segformer3d
  epochs: 100
  batch_size: 2
  use_amp: true

models:                    # Model-specific hyperparameters
  segresnet:
    middle:
      learning_rate: 1.0e-4
      lr_decay_rate: 0.75
      freeze_encoder_ratio: 0.5
```

---

## 🚀 Kullanım

### Eğitim Başlatma
```bash
python train.py --model segresnet --tune middle --epochs 100
```

### Eğitimi Devam Ettirme
```bash
python train.py --resume runs/segresnet_2024-02-01_12-00-00
```

### LMDB Oluşturma
```bash
python -m src.data.lmdb_converter --csv-file dataset/raw/train.csv --overwrite
```

---

## 💻 Gereksinimler

| Kaynak | Minimum | Önerilen |
|--------|---------|----------|
| **GPU VRAM** | 8 GB | 16+ GB |
| **System RAM** | 16 GB | 32+ GB |
| **Storage** | 50 GB SSD | 100+ GB NVMe |

> ⚠️ **Not:** 3D volumetrik veri (256³) 2D görüntülerden ~100x daha büyüktür. MacBook gibi cihazlarda eğitim zordur, cloud GPU önerilir.

---

## 📦 Bağımlılıklar

```
torch>=2.0
monai>=1.3
lmdb
tifffile
numpy
pandas
scipy
tqdm
matplotlib
```

---

## 📊 Eğitim Çıktıları

Her eğitim `runs/` altında benzersiz bir klasör oluşturur:

```
runs/segresnet_2024-02-01_12-00-00/
├── config.yaml           # Kullanılan konfigürasyon
├── train.log             # Eğitim logları
├── models/
│   ├── best.pth          # En iyi model
│   └── last.pth          # Son checkpoint
├── metrics/
│   ├── train_metrics.csv # Epoch başına train metrikleri
│   └── val_metrics.csv   # Epoch başına val metrikleri
└── plots/
    ├── loss.png          # Loss eğrisi
    └── metrics.png       # Metrik eğrileri
```

---

## 🏆 Yarışma Hakkında

[Vesuvius Challenge](https://scrollprize.org) - Herculaneum'da volkanik kül altında korunan antik parşömenleri okumayı amaçlayan bir yarışma. Bu proje, parşömen yüzeylerini 3D CT taramalarından tespit eden segmentasyon modeli geliştirmektedir.

---

## 📄 Lisans

MIT License
