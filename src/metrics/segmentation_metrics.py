"""
Standard segmentation metrics.

Provides common voxel-wise metrics for binary segmentation:
- Dice coefficient
- IoU (Intersection over Union / Jaccard Index)
- Precision, Recall, F1
- Accuracy
"""

import numpy as np


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient (F1 score for binary segmentation).
    
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    
    intersection = np.sum(pred & gt)
    union_sum = np.sum(pred) + np.sum(gt)
    
    if union_sum == 0:
        return 1.0  # Both empty
    
    dice = (2.0 * intersection + smooth) / (union_sum + smooth)
    return float(dice)


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute IoU (Intersection over Union / Jaccard Index).
    
    IoU = |P ∩ G| / |P ∪ G|
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    
    if union == 0:
        return 1.0  # Both empty
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def precision_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute precision (positive predictive value).
    
    Precision = TP / (TP + FP)
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Precision in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    
    if tp + fp == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return float(precision)


def recall_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute recall (sensitivity, true positive rate).
    
    Recall = TP / (TP + FN)
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Recall in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    
    tp = np.sum(pred & gt)
    fn = np.sum(~pred & gt)
    
    if tp + fn == 0:
        return 1.0  # No ground truth positives
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(recall)


def f1_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * Precision * Recall / (Precision + Recall)
    
    Note: This is equivalent to Dice coefficient for binary segmentation.
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        smooth: Smoothing factor
        
    Returns:
        F1 score in [0, 1]
    """
    prec = precision_score(pred, gt, smooth)
    rec = recall_score(pred, gt, smooth)
    
    if prec + rec == 0:
        return 0.0
    
    f1 = 2 * prec * rec / (prec + rec)
    return float(f1)


def accuracy_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute voxel-wise accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        
    Returns:
        Accuracy in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    
    correct = np.sum(pred == gt)
    total = len(pred)
    
    if total == 0:
        return 1.0
    
    return float(correct / total)
