"""
Competition metrics for Vesuvius Challenge leaderboard.

Combines the three main metrics:
- SurfaceDice@τ (35%)
- VOI_score (35%)  
- TopoScore (30%)

Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score
"""

from typing import Dict, Tuple

import numpy as np

from .surface_dice import surface_dice
from .voi_score import voi_score
from .topo_score import topo_score
from .segmentation_metrics import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def competition_score(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    voi_alpha: float = 0.3,
    topo_weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
    weights: Tuple[float, float, float] = (0.30, 0.35, 0.35)
) -> float:
    """
    Compute the competition leaderboard score.
    
    Score = w_topo × TopoScore + w_sdice × SurfaceDice@τ + w_voi × VOI_score
    
    Default weights: (0.30, 0.35, 0.35) for (TopoScore, SurfaceDice, VOI)
    
    Args:
        pred: Binary prediction mask (D, H, W)
        gt: Binary ground truth mask (D, H, W)
        tau: Tolerance for Surface Dice (default 2.0)
        spacing: Voxel spacing (sz, sy, sx)
        voi_alpha: Alpha parameter for VOI score
        topo_weights: Per-dimension weights for TopoScore
        weights: Weights for (TopoScore, SurfaceDice, VOI)
        
    Returns:
        Competition score in [0, 1]
    """
    # Compute individual metrics
    sdice = surface_dice(pred, gt, tau=tau, spacing=spacing)
    voi, _, _ = voi_score(pred, gt, alpha=voi_alpha)
    topo, _ = topo_score(pred, gt, weights=topo_weights)
    
    # Weighted combination
    w_topo, w_sdice, w_voi = weights
    score = w_topo * topo + w_sdice * sdice + w_voi * voi
    
    return float(score)


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    voi_alpha: float = 0.3,
    topo_weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
    competition_weights: Tuple[float, float, float] = (0.30, 0.35, 0.35)
) -> Dict[str, float]:
    """
    Compute all metrics for a prediction-ground truth pair.
    
    Returns a dictionary with:
    - Competition metrics: surface_dice, voi_score, topo_score, competition_score
    - Standard metrics: dice, iou, precision, recall, accuracy
    
    Args:
        pred: Binary prediction mask (D, H, W)
        gt: Binary ground truth mask (D, H, W)
        tau: Tolerance for Surface Dice
        spacing: Voxel spacing (sz, sy, sx)
        voi_alpha: Alpha parameter for VOI score
        topo_weights: Per-dimension weights for TopoScore
        competition_weights: Weights for final score
        
    Returns:
        Dictionary with all metric values
    """
    # Competition metrics
    sdice = surface_dice(pred, gt, tau=tau, spacing=spacing)
    voi, voi_split, voi_merge = voi_score(pred, gt, alpha=voi_alpha)
    topo, topo_per_dim = topo_score(pred, gt, weights=topo_weights)
    
    # Standard metrics
    dice = dice_coefficient(pred, gt)
    iou = iou_score(pred, gt)
    prec = precision_score(pred, gt)
    rec = recall_score(pred, gt)
    acc = accuracy_score(pred, gt)
    
    # Competition score
    w_topo, w_sdice, w_voi = competition_weights
    comp_score = w_topo * topo + w_sdice * sdice + w_voi * voi
    
    return {
        # Competition metrics
        "competition_score": comp_score,
        "surface_dice": sdice,
        "voi_score": voi,
        "voi_split": voi_split,
        "voi_merge": voi_merge,
        "topo_score": topo,
        "topo_f1_dim0": topo_per_dim.get(0, 1.0),
        "topo_f1_dim1": topo_per_dim.get(1, 1.0),
        "topo_f1_dim2": topo_per_dim.get(2, 1.0),
        # Standard metrics
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
    }
