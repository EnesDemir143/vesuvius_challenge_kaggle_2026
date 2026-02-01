"""
Trainer class for Vesuvius Challenge Surface Detection.

Features:
- Training and validation loops
- Checkpoint saving (last.pth, best.pth)
- Resume from checkpoint
- Mixed precision training (AMP)
- Gradient accumulation
- Early stopping
- Validation probability saving
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .experiment_logger import ExperimentLogger
from ..metrics import compute_all_metrics


class Trainer:
    """
    Trainer for 3D surface segmentation models.
    
    Handles training loop, validation, checkpointing, and resume functionality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: Callable,
        logger: ExperimentLogger,
        device: torch.device,
        # Training settings
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        clip_grad_norm: Optional[float] = 1.0,
        # Checkpoint settings
        save_every_n_epochs: int = 5,
        save_probs_every_n_epochs: int = 5,
        keep_last_n_prob_epochs: int = 3,
        # Early stopping
        early_stopping_patience: int = 20,
        early_stopping_metric: str = "competition_score",
        early_stopping_mode: str = "max",
        # Metrics settings
        metrics_tau: float = 2.0,
        metrics_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function
            logger: ExperimentLogger instance
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            grad_accum_steps: Number of gradient accumulation steps
            clip_grad_norm: Gradient clipping norm (None to disable)
            save_every_n_epochs: Save checkpoint every N epochs
            save_probs_every_n_epochs: Save validation probs every N epochs
            keep_last_n_prob_epochs: Number of probability epochs to keep
            early_stopping_patience: Epochs to wait for improvement
            early_stopping_metric: Metric to monitor for early stopping
            early_stopping_mode: 'max' or 'min'
            metrics_tau: Tolerance for Surface Dice
            metrics_spacing: Voxel spacing for metrics
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device
        
        # Training settings
        self.use_amp = use_amp and device.type == 'cuda'
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Checkpoint settings
        self.save_every_n_epochs = save_every_n_epochs
        self.save_probs_every_n_epochs = save_probs_every_n_epochs
        self.keep_last_n_prob_epochs = keep_last_n_prob_epochs
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.best_metric = float('-inf') if early_stopping_mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics settings
        self.metrics_tau = metrics_tau
        self.metrics_spacing = metrics_spacing
        
        # State
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Load data to device with proper dtype (float16/int8 in RAM, convert to float32/long on GPU)
            images = batch['image'].to(self.device, dtype=torch.float32)
            labels = batch['label'].to(self.device, dtype=torch.long)
            
            # Add channel dimension if needed
            if images.dim() == 4:  # (B, D, H, W)
                images = images.unsqueeze(1)  # (B, 1, D, H, W)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        # Get current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'lr': lr,
        }
    
    @torch.no_grad()
    def validate_epoch(self, save_probs: bool = False) -> Dict[str, float]:
        """
        Run validation epoch.
        
        Args:
            save_probs: Whether to save prediction probabilities
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}
        num_samples = 0
        
        # Setup prob saving
        prob_dir = None
        if save_probs:
            prob_dir = self.logger.get_prob_epoch_dir(self.current_epoch)
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch} [Val]",
            leave=False
        )
        
        for batch in pbar:
            # Load data to device with proper dtype (float16/int8 in RAM, convert to float32/long on GPU)
            images = batch['image'].to(self.device, dtype=torch.float32)
            labels = batch['label'].to(self.device, dtype=torch.long)
            vol_ids = batch.get('id', [f'sample_{i}' for i in range(len(images))])
            
            # Add channel dimension if needed
            if images.dim() == 4:
                images = images.unsqueeze(1)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            if outputs.shape[1] > 1:
                # Multi-class: softmax + argmax
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
            else:
                # Binary: sigmoid + threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().squeeze(1)
            
            # Compute metrics for each sample
            for i in range(len(images)):
                pred_np = preds[i].cpu().numpy()
                gt_np = labels[i].cpu().numpy()
                
                # Binarize (foreground = 1)
                pred_binary = (pred_np == 1).astype(np.uint8)
                gt_binary = (gt_np == 1).astype(np.uint8)
                
                # Compute all metrics
                metrics = compute_all_metrics(
                    pred_binary, gt_binary,
                    tau=self.metrics_tau,
                    spacing=self.metrics_spacing
                )
                
                for name, value in metrics.items():
                    if name not in all_metrics:
                        all_metrics[name] = []
                    all_metrics[name].append(value)
                
                # Save probabilities
                if save_probs and prob_dir:
                    vol_id = vol_ids[i] if isinstance(vol_ids, list) else vol_ids
                    prob_np = probs[i].cpu().numpy()
                    np.save(prob_dir / f"{vol_id}.npy", prob_np)
                
                num_samples += 1
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / max(len(self.val_loader), 1)
        }
        for name, values in all_metrics.items():
            avg_metrics[name] = np.mean(values)
        
        return avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_metric': self.best_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
        }
        
        # Save last checkpoint
        last_path = self.logger.models_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.logger.models_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved! {self.early_stopping_metric}: {self.best_metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file or run directory
            
        Returns:
            Start epoch (1-indexed, i.e., the next epoch to train)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # If directory, look for last.pth
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'models' / 'last.pth'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore scaler
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resuming from epoch {start_epoch}")
        
        return start_epoch
    
    def train(
        self,
        epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Args:
            epochs: Total number of epochs to train
            start_epoch: Starting epoch (1-indexed)
            
        Returns:
            Dictionary with training history and best metrics
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training: epochs {start_epoch} to {epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info("=" * 60)
        
        best_epoch = start_epoch
        
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            save_probs = (epoch % self.save_probs_every_n_epochs == 0)
            val_metrics = self.validate_epoch(save_probs=save_probs)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self.logger.log_metrics('train_metrics', epoch, train_metrics)
            self.logger.log_metrics('val_metrics', epoch, val_metrics)
            
            # Check for improvement
            current_metric = val_metrics.get(self.early_stopping_metric, 0)
            
            if self.early_stopping_mode == 'max':
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                best_epoch = epoch
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % self.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Cleanup old prob directories
            if save_probs:
                self.logger.cleanup_old_prob_dirs(
                    epoch, 
                    keep_last_n=self.keep_last_n_prob_epochs
                )
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Competition: {val_metrics.get('competition_score', 0):.4f} | "
                f"Best: {self.best_metric:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break
        
        # Generate plots
        self.logger.info("Generating plots...")
        self.logger.plot_metrics()
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete!")
        self.logger.info(f"Best {self.early_stopping_metric}: {self.best_metric:.4f} at epoch {best_epoch}")
        self.logger.info(f"Run directory: {self.logger.run_dir}")
        self.logger.info("=" * 60)
        
        return {
            'best_metric': self.best_metric,
            'best_epoch': best_epoch,
            'final_epoch': self.current_epoch,
        }
