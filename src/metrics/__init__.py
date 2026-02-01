"""
Metrics for Vesuvius Challenge 3D Surface Segmentation.

Competition Metrics:
- SurfaceDice@τ: Surface-aware Dice with tolerance
- VOI_score: Variation of Information for instance consistency
- TopoScore: Betti number matching for topological correctness

Standard Metrics:
- Dice coefficient
- IoU (Intersection over Union)
- Precision, Recall, F1
"""

from .surface_dice import surface_dice, extract_surface
from .voi_score import voi_score, compute_voi
from .topo_score import topo_score, compute_betti_numbers
from .segmentation_metrics import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from .competition_metrics import competition_score, compute_all_metrics

__all__ = [
    # Competition metrics
    "surface_dice",
    "voi_score", 
    "topo_score",
    "competition_score",
    "compute_all_metrics",
    # Standard metrics
    "dice_coefficient",
    "iou_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "accuracy_score",
    # Utility functions
    "extract_surface",
    "compute_voi",
    "compute_betti_numbers",
]
