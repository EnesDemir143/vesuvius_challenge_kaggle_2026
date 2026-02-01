"""
Augmentation transforms for 3D volumetric data.

Pipeline Order (Training):
1. ZJitter: Z-axis random shift
2. BasicAugs: HorizontalFlip, VerticalFlip, RandomRotate90
3. PipMix: Mix two volumes (batch-level, called separately)
4. Normalize: Mean/std normalization
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


class CenterCropOrPad:
    """
    Center crop or pad volume to target size.
    
    If the volume is larger than target, center crop.
    If the volume is smaller than target, zero-pad.
    
    Args:
        target_size: Target size as (D, H, W) tuple or single int for cubic
    """
    
    def __init__(self, target_size: Union[int, Tuple[int, int, int]] = 256):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size, target_size)
        else:
            self.target_size = target_size
    
    def __call__(
        self, 
        image: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply center crop or padding.
        
        Args:
            image: (D, H, W) tensor
            label: Optional (D, H, W) tensor
            
        Returns:
            Cropped/padded image and optionally label
        """
        current_size = image.shape
        target_d, target_h, target_w = self.target_size
        
        # Process each dimension
        image = self._process_dim(image, 0, current_size[0], target_d)
        image = self._process_dim(image, 1, image.shape[1], target_h)
        image = self._process_dim(image, 2, image.shape[2], target_w)
        
        if label is not None:
            label = self._process_dim(label, 0, current_size[0], target_d, pad_value=2)
            label = self._process_dim(label, 1, label.shape[1], target_h, pad_value=2)
            label = self._process_dim(label, 2, label.shape[2], target_w, pad_value=2)
            return image, label
        
        return image
    
    def _process_dim(
        self, 
        tensor: torch.Tensor, 
        dim: int, 
        current: int, 
        target: int,
        pad_value: float = 0
    ) -> torch.Tensor:
        """Process a single dimension: crop if larger, pad if smaller."""
        if current == target:
            return tensor
        elif current > target:
            # Center crop
            start = (current - target) // 2
            indices = [slice(None)] * 3
            indices[dim] = slice(start, start + target)
            return tensor[tuple(indices)]
        else:
            # Zero pad
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_list = [0, 0, 0, 0, 0, 0]  # (w_before, w_after, h_before, h_after, d_before, d_after)
            # torch.nn.functional.pad uses reverse order
            pad_idx = (2 - dim) * 2
            pad_list[pad_idx] = pad_before
            pad_list[pad_idx + 1] = pad_after
            return torch.nn.functional.pad(tensor, pad_list, value=pad_value)


class ZJitter:
    """
    Z-axis jitter: randomly shift layers along Z-axis.
    
    Args:
        jitter_range: Maximum shift amount in slices (±jitter_range)
    """
    
    def __init__(self, jitter_range: int = 5):
        self.jitter_range = jitter_range
    
    def __call__(
        self, 
        image: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply Z-axis jitter.
        
        Args:
            image: (D, H, W) tensor
            label: Optional (D, H, W) tensor
            
        Returns:
            Jittered image and optionally label
        """
        if self.jitter_range == 0:
            return (image, label) if label is not None else image
            
        shift = np.random.randint(-self.jitter_range, self.jitter_range + 1)
        
        if shift == 0:
            return (image, label) if label is not None else image
        
        # Roll along Z-axis (dim 0)
        image = torch.roll(image, shifts=shift, dims=0)
        
        # Zero out wrapped edges
        if shift > 0:
            image[:shift] = 0
        else:
            image[shift:] = 0
        
        if label is not None:
            label = torch.roll(label, shifts=shift, dims=0)
            if shift > 0:
                label[:shift] = 2  # Unlabeled value
            else:
                label[shift:] = 2
            return image, label
        
        return image


class BasicAugs:
    """
    Basic geometric augmentations: HorizontalFlip, VerticalFlip, RandomRotate90.
    
    Args:
        p_hflip: Probability of horizontal flip
        p_vflip: Probability of vertical flip
        p_rot90: Probability of 90° rotation
    """
    
    def __init__(
        self, 
        p_hflip: float = 0.5, 
        p_vflip: float = 0.5, 
        p_rot90: float = 0.5
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
    
    def __call__(
        self, 
        image: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply basic augmentations.
        
        Args:
            image: (D, H, W) tensor
            label: Optional (D, H, W) tensor
            
        Returns:
            Augmented image and optionally label
        """
        # Horizontal flip (along W axis, dim 2)
        if np.random.random() < self.p_hflip:
            image = torch.flip(image, dims=[2])
            if label is not None:
                label = torch.flip(label, dims=[2])
        
        # Vertical flip (along H axis, dim 1)
        if np.random.random() < self.p_vflip:
            image = torch.flip(image, dims=[1])
            if label is not None:
                label = torch.flip(label, dims=[1])
        
        # Random 90° rotation (in H-W plane)
        if np.random.random() < self.p_rot90:
            k = np.random.randint(1, 4)  # 90°, 180°, or 270°
            image = torch.rot90(image, k=k, dims=[1, 2])
            if label is not None:
                label = torch.rot90(label, k=k, dims=[1, 2])
        
        return (image, label) if label is not None else image


class PipMix:
    """
    PipMix: Mix two volumes by cutting and pasting regions.
    
    This is a batch-level augmentation that should be called 
    on pairs of samples from the dataset.
    
    Args:
        alpha: Beta distribution parameter for lambda sampling
        p: Probability of applying PipMix
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        image1: torch.Tensor,
        label1: torch.Tensor,
        image2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Mix two volumes using CutMix-style approach for 3D data.
        
        Args:
            image1, label1: First sample (D, H, W)
            image2, label2: Second sample (D, H, W)
            
        Returns:
            Mixed image, mixed label, lambda value
        """
        if np.random.random() > self.p:
            return image1, label1, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        D, H, W = image1.shape
        
        # Calculate cut size based on lambda
        cut_ratio = np.sqrt(1 - lam)
        cut_d = int(D * cut_ratio)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # Random cut position
        d1 = np.random.randint(0, D - cut_d + 1) if cut_d < D else 0
        h1 = np.random.randint(0, H - cut_h + 1) if cut_h < H else 0
        w1 = np.random.randint(0, W - cut_w + 1) if cut_w < W else 0
        
        d2, h2, w2 = d1 + cut_d, h1 + cut_h, w1 + cut_w
        
        # Create mixed samples
        mixed_image = image1.clone()
        mixed_label = label1.clone()
        
        mixed_image[d1:d2, h1:h2, w1:w2] = image2[d1:d2, h1:h2, w1:w2]
        mixed_label[d1:d2, h1:h2, w1:w2] = label2[d1:d2, h1:h2, w1:w2]
        
        # Recalculate lambda based on actual cut
        lam = 1 - (cut_d * cut_h * cut_w) / (D * H * W)
        
        return mixed_image, mixed_label, lam


class Normalize:
    """
    Normalize with mean and std.
    
    Args:
        mean: Mean value(s)
        std: Std value(s)
        stats_path: Path to stats file (mean_std.npy)
    """
    
    def __init__(
        self, 
        mean: Optional[float] = None, 
        std: Optional[float] = None,
        stats_path: Optional[Union[str, Path]] = None
    ):
        if stats_path is not None:
            stats = np.load(stats_path, allow_pickle=True).item()
            self.mean = float(stats['mean'][0])
            self.std = float(stats['std'][0])
        elif mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            # Default fallback values for uint8 data
            self.mean = 127.5
            self.std = 127.5
    
    def __call__(
        self, 
        image: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalize image.
        
        Args:
            image: (D, H, W) tensor
            label: Optional (D, H, W) tensor (not modified)
            
        Returns:
            Normalized image and optionally label
        """
        image = (image - self.mean) / self.std
        
        return (image, label) if label is not None else image


class AugmentationPipeline:
    """
    Complete augmentation pipeline for training and validation.
    
    Training: CenterCropOrPad → ZJitter → BasicAugs → Normalize
    Validation: CenterCropOrPad → Normalize
    
    Note: PipMix is batch-level and should be applied separately in training loop.
    
    Args:
        mode: 'train' or 'val'
        target_size: Target volume size (default 256)
        z_jitter_range: Max Z-axis shift
        stats_path: Path to mean_std.npy
        mean: Manual mean value (if stats_path not provided)
        std: Manual std value (if stats_path not provided)
    """
    
    def __init__(
        self,
        mode: str = 'train',
        target_size: Union[int, Tuple[int, int, int]] = 256,
        z_jitter_range: int = 5,
        stats_path: Optional[Union[str, Path]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ):
        self.mode = mode
        
        # Initialize transforms
        self.crop_or_pad = CenterCropOrPad(target_size=target_size)
        self.z_jitter = ZJitter(jitter_range=z_jitter_range) if mode == 'train' else None
        self.basic_augs = BasicAugs() if mode == 'train' else None
        self.normalize = Normalize(mean=mean, std=std, stats_path=stats_path)
    
    def __call__(
        self, 
        image: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply augmentation pipeline.
        
        Args:
            image: (D, H, W) tensor
            label: Optional (D, H, W) tensor
            
        Returns:
            Augmented/normalized image and optionally label
        """
        # Step 0: Center crop or pad to target size (always applied)
        if label is not None:
            image, label = self.crop_or_pad(image, label)
        else:
            image = self.crop_or_pad(image)
        
        if self.mode == 'train':
            # Step 1: Z-Jitter
            if label is not None:
                image, label = self.z_jitter(image, label)
            else:
                image = self.z_jitter(image)
            
            # Step 2: Basic Augmentations
            if label is not None:
                image, label = self.basic_augs(image, label)
            else:
                image = self.basic_augs(image)
        
        # Step 3 (or only step for val): Normalize
        if label is not None:
            image, label = self.normalize(image, label)
            return image, label
        else:
            return self.normalize(image)


def get_train_transforms(
    stats_path: Optional[Union[str, Path]] = None,
    z_jitter_range: int = 5,
    target_size: Union[int, Tuple[int, int, int]] = 256
) -> AugmentationPipeline:
    """Get training augmentation pipeline."""
    return AugmentationPipeline(
        mode='train',
        target_size=target_size,
        z_jitter_range=z_jitter_range,
        stats_path=stats_path
    )


def get_val_transforms(
    stats_path: Optional[Union[str, Path]] = None,
    target_size: Union[int, Tuple[int, int, int]] = 256
) -> AugmentationPipeline:
    """Get validation augmentation pipeline (normalize only)."""
    return AugmentationPipeline(
        mode='val',
        target_size=target_size,
        stats_path=stats_path
    )


def get_pipmix() -> PipMix:
    """Get PipMix transform for batch-level augmentation."""
    return PipMix(alpha=1.0, p=0.5)
