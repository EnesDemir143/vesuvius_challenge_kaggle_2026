"""
Surface Dice metric with tolerance τ.

Surface Dice measures the fraction of surface points (both prediction and ground truth)
that lie within a spatial tolerance τ of each other.
"""

from typing import Tuple

import numpy as np
from scipy import ndimage


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """
    Extract the surface voxels from a binary mask.
    
    Surface voxels are foreground voxels that have at least one 
    background neighbor (using 6-connectivity for efficiency).
    
    Args:
        mask: Binary mask (D, H, W)
        
    Returns:
        Binary surface mask (D, H, W)
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Erode the mask
    eroded = ndimage.binary_erosion(mask)
    
    # Surface = original - eroded
    surface = mask.astype(bool) & ~eroded
    
    return surface


def compute_surface_distances(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Compute distances from surface_a points to nearest surface_b points.
    
    Args:
        surface_a: Binary surface mask (D, H, W)
        surface_b: Binary surface mask (D, H, W)
        spacing: Voxel spacing (sz, sy, sx)
        
    Returns:
        Array of distances for each surface_a point
    """
    if surface_a.sum() == 0:
        return np.array([])
    
    if surface_b.sum() == 0:
        # No surface_b points, return infinite distances
        return np.full(int(surface_a.sum()), np.inf)
    
    # Compute distance transform from surface_b
    # distance_transform_edt gives distance from background to nearest foreground
    distance_map = ndimage.distance_transform_edt(~surface_b, sampling=spacing)
    
    # Get distances for surface_a points
    distances = distance_map[surface_a]
    
    return distances


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Surface Dice with tolerance τ.
    
    Surface Dice measures the fraction of surface points that are within
    tolerance τ of the other surface. It's tolerant of slight boundary
    misplacements while penalizing larger errors.
    
    Args:
        pred: Binary prediction mask (D, H, W)
        gt: Binary ground truth mask (D, H, W)
        tau: Tolerance in physical units (same units as spacing)
        spacing: Voxel spacing (sz, sy, sx) in physical units
        
    Returns:
        Surface Dice score in [0, 1]
        
    Edge cases:
        - Both empty: returns 1.0
        - One empty, other not: returns 0.0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0
    
    # Handle edge cases
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    
    # Extract surfaces
    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)
    
    # Handle case where surfaces are empty (solid blocks)
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0
    
    # Compute distances: pred_surface -> gt_surface
    dist_pred_to_gt = compute_surface_distances(pred_surface, gt_surface, spacing)
    
    # Compute distances: gt_surface -> pred_surface  
    dist_gt_to_pred = compute_surface_distances(gt_surface, pred_surface, spacing)
    
    # Count matches within tolerance
    pred_matches = np.sum(dist_pred_to_gt <= tau)
    gt_matches = np.sum(dist_gt_to_pred <= tau)
    
    # Total surface points
    n_pred = len(dist_pred_to_gt)
    n_gt = len(dist_gt_to_pred)
    
    # Surface Dice = 2 * (matched / total) averaged both ways
    # = (pred_matches + gt_matches) / (n_pred + n_gt)
    surface_dice_score = (pred_matches + gt_matches) / (n_pred + n_gt)
    
    return float(surface_dice_score)
