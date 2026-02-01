"""
Variation of Information (VOI) score for instance consistency.

VOI compares connected-component labelings of prediction and ground truth,
decomposing into VOI_split (over-segmentation) and VOI_merge (under-segmentation).
"""

from typing import Tuple

import numpy as np
from scipy import ndimage


def compute_contingency_table(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the contingency table between two labelings.
    
    Args:
        pred_labels: Integer label array from prediction
        gt_labels: Integer label array from ground truth
        
    Returns:
        (contingency_table, pred_counts, gt_counts)
    """
    # Get unique labels (excluding 0 = background)
    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)
    
    pred_unique = pred_unique[pred_unique > 0]
    gt_unique = gt_unique[gt_unique > 0]
    
    if len(pred_unique) == 0 or len(gt_unique) == 0:
        return np.array([[]]), np.array([]), np.array([])
    
    # Create label mappings for indexing
    pred_map = {label: i for i, label in enumerate(pred_unique)}
    gt_map = {label: i for i, label in enumerate(gt_unique)}
    
    # Initialize contingency table
    n_pred = len(pred_unique)
    n_gt = len(gt_unique)
    contingency = np.zeros((n_pred, n_gt), dtype=np.int64)
    
    # Fill contingency table
    mask = (pred_labels > 0) & (gt_labels > 0)
    for p, g in zip(pred_labels[mask], gt_labels[mask]):
        contingency[pred_map[p], gt_map[g]] += 1
    
    # Compute marginals
    pred_counts = contingency.sum(axis=1)
    gt_counts = contingency.sum(axis=0)
    
    return contingency, pred_counts, gt_counts


def compute_voi(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute Variation of Information between two labelings.
    
    VOI = VOI_split + VOI_merge
    - VOI_split = H(GT | Pred): measures over-segmentation (splits)
    - VOI_merge = H(Pred | GT): measures under-segmentation (merges)
    
    Args:
        pred_labels: Integer label array from prediction
        gt_labels: Integer label array from ground truth
        
    Returns:
        (voi_total, voi_split, voi_merge)
    """
    contingency, pred_counts, gt_counts = compute_contingency_table(pred_labels, gt_labels)
    
    if contingency.size == 0 or contingency.sum() == 0:
        return 0.0, 0.0, 0.0
    
    n_total = contingency.sum()
    
    # Compute entropies using the contingency table
    # H(GT | Pred) = H(GT, Pred) - H(Pred)
    # H(Pred | GT) = H(GT, Pred) - H(GT)
    
    # Joint entropy H(GT, Pred)
    p_joint = contingency / n_total
    p_joint = p_joint[p_joint > 0]
    h_joint = -np.sum(p_joint * np.log2(p_joint))
    
    # H(Pred)
    p_pred = pred_counts / n_total
    p_pred = p_pred[p_pred > 0]
    h_pred = -np.sum(p_pred * np.log2(p_pred))
    
    # H(GT)
    p_gt = gt_counts / n_total
    p_gt = p_gt[p_gt > 0]
    h_gt = -np.sum(p_gt * np.log2(p_gt))
    
    # Conditional entropies
    voi_split = h_joint - h_pred  # H(GT | Pred)
    voi_merge = h_joint - h_gt    # H(Pred | GT)
    voi_total = voi_split + voi_merge
    
    return float(voi_total), float(voi_split), float(voi_merge)


def voi_score(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.3,
    connectivity: int = 26
) -> Tuple[float, float, float]:
    """
    Compute VOI score from binary masks.
    
    Converts binary masks to connected component labelings, then computes VOI.
    The VOI is converted to a bounded score: VOI_score = 1 / (1 + α * VOI_total)
    
    Args:
        pred: Binary prediction mask (D, H, W)
        gt: Binary ground truth mask (D, H, W)
        alpha: Scaling factor for VOI_total (default 0.3)
        connectivity: Connectivity for connected components (6, 18, or 26)
        
    Returns:
        (voi_score, voi_split, voi_merge) where voi_score is in [0, 1]
        
    Edge cases:
        - Both empty: returns (1.0, 0.0, 0.0)
        - One empty, other not: returns near 0 score
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0
    
    # Handle edge cases
    if pred_empty and gt_empty:
        return 1.0, 0.0, 0.0
    
    # Create structure for connectivity
    if connectivity == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    else:  # 26
        structure = ndimage.generate_binary_structure(3, 3)
    
    # Compute connected components
    pred_labels, n_pred = ndimage.label(pred, structure=structure)
    gt_labels, n_gt = ndimage.label(gt, structure=structure)
    
    # Handle case where one is empty
    if n_pred == 0 or n_gt == 0:
        # Maximum penalty - use a heuristic
        return 0.01, float('inf'), float('inf')
    
    # Compute VOI
    voi_total, voi_split, voi_merge = compute_voi(pred_labels, gt_labels)
    
    # Convert to bounded score
    score = 1.0 / (1.0 + alpha * voi_total)
    
    return float(score), float(voi_split), float(voi_merge)
