"""
TopoScore: Topological correctness using Betti number matching.

Uses persistent homology (via GUDHI) to compare topological features:
- k=0: connected components
- k=1: tunnels/handles  
- k=2: cavities/voids
"""

from typing import Dict, List, Tuple

import numpy as np

# Try to import gudhi, provide fallback
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def compute_betti_numbers(mask: np.ndarray, max_dimension: int = 2) -> Dict[int, int]:
    """
    Compute Betti numbers for a binary 3D mask using cubical complex.
    
    Betti numbers count topological features:
    - β₀: number of connected components
    - β₁: number of tunnels/handles (1D holes)
    - β₂: number of cavities/voids (2D holes)
    
    Args:
        mask: Binary mask (D, H, W)
        max_dimension: Maximum homology dimension to compute
        
    Returns:
        Dictionary mapping dimension to Betti number {0: β₀, 1: β₁, 2: β₂}
    """
    if not GUDHI_AVAILABLE:
        # Fallback: use connected components for β₀ only
        from scipy import ndimage
        structure = ndimage.generate_binary_structure(3, 3)
        _, n_components = ndimage.label(mask.astype(bool), structure=structure)
        return {0: n_components, 1: 0, 2: 0}
    
    mask = mask.astype(bool)
    
    if mask.sum() == 0:
        return {k: 0 for k in range(max_dimension + 1)}
    
    # Create cubical complex
    # GUDHI expects values where lower = foreground
    # We invert: foreground=0, background=1
    cubical_data = (~mask).astype(np.float64)
    
    # Create cubical complex - use bitmap_file approach or top_dimensional_cells
    # top_dimensional_cells should be a flattened array with dimensions as list
    cc = gudhi.CubicalComplex(
        top_dimensional_cells=cubical_data.flatten().tolist(),
        dimensions=list(mask.shape)
    )
    
    # Compute persistence
    cc.compute_persistence()
    
    # Get Betti numbers
    betti = cc.betti_numbers()
    
    # Convert to dict with default 0 for missing dimensions
    result = {}
    for k in range(max_dimension + 1):
        result[k] = betti[k] if k < len(betti) else 0
    
    return result


def topological_f1(pred_betti: int, gt_betti: int) -> float:
    """
    Compute Topological F1 for a single dimension.
    
    Uses a matching-based approach where:
    - TP = min(pred, gt) (matched features)
    - FP = max(0, pred - gt) (extra predicted features)
    - FN = max(0, gt - pred) (missing ground truth features)
    
    F1 = 2*TP / (2*TP + FP + FN)
    
    Args:
        pred_betti: Predicted Betti number
        gt_betti: Ground truth Betti number
        
    Returns:
        F1 score in [0, 1]
    """
    if pred_betti == 0 and gt_betti == 0:
        return 1.0  # Perfect match for empty
    
    tp = min(pred_betti, gt_betti)
    fp = max(0, pred_betti - gt_betti)
    fn = max(0, gt_betti - pred_betti)
    
    if tp == 0:
        return 0.0
    
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


def topo_score(
    pred: np.ndarray,
    gt: np.ndarray,
    weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
    max_dimension: int = 2
) -> Tuple[float, Dict[int, float]]:
    """
    Compute TopoScore using Betti number matching.
    
    Compares topological features in each homology dimension and returns
    a weighted average of per-dimension Topological-F1 scores.
    
    Args:
        pred: Binary prediction mask (D, H, W)
        gt: Binary ground truth mask (D, H, W)
        weights: Weights for dimensions (w0, w1, w2)
        max_dimension: Maximum homology dimension
        
    Returns:
        (topo_score, per_dimension_f1) where topo_score is in [0, 1]
        
    Edge cases:
        - Both empty: returns (1.0, {})
        - One empty, other not: returns (0.0 for k=0, ...)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0
    
    # Handle edge cases
    if pred_empty and gt_empty:
        return 1.0, {k: 1.0 for k in range(max_dimension + 1)}
    
    # Compute Betti numbers
    pred_betti = compute_betti_numbers(pred, max_dimension)
    gt_betti = compute_betti_numbers(gt, max_dimension)
    
    # Compute per-dimension F1
    per_dim_f1 = {}
    active_dims = []
    active_weights = []
    
    for k in range(max_dimension + 1):
        p_k = pred_betti.get(k, 0)
        g_k = gt_betti.get(k, 0)
        
        # Skip dimensions with no features in both
        if p_k == 0 and g_k == 0:
            per_dim_f1[k] = 1.0  # Perfect match for empty dimension
            continue
        
        f1_k = topological_f1(p_k, g_k)
        per_dim_f1[k] = f1_k
        active_dims.append(k)
        active_weights.append(weights[k] if k < len(weights) else 0.0)
    
    # Compute weighted average with renormalization
    if len(active_dims) == 0:
        return 1.0, per_dim_f1
    
    total_weight = sum(active_weights)
    if total_weight == 0:
        return 1.0, per_dim_f1
    
    score = sum(
        per_dim_f1[k] * (weights[k] if k < len(weights) else 0.0) 
        for k in active_dims
    ) / total_weight
    
    return float(score), per_dim_f1
