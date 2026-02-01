"""
Experiment Logger for managing runs, logging, and visualization.

Features:
- Run folder creation with timestamps
- CSV logging for train/val metrics
- Plot generation for loss, learning rate, and metrics
- Resume support (append to existing CSVs)
"""

import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import yaml


class ExperimentLogger:
    """
    Manages experiment logging, file organization, and visualization.
    
    Creates and manages run folder structure:
        runs/{model_name}_{timestamp}/
        ├── config.yaml
        ├── train.log
        ├── train_metrics.csv
        ├── val_metrics.csv
        ├── system_stats.csv
        ├── figs/
        │   ├── loss.png
        │   ├── learning_rate.png
        │   └── metrics.png
        ├── models/
        │   ├── last.pth
        │   └── best.pth
        └── probs/
            └── epoch_{N}/
                └── {volume_id}.npy
    """
    
    def __init__(
        self,
        run_dir: Optional[Union[str, Path]] = None,
        model_name: str = "model",
        config: Optional[Dict] = None,
        resume: bool = False,
        runs_base_dir: str = "runs"
    ):
        """
        Initialize ExperimentLogger.
        
        Args:
            run_dir: Existing run directory to resume from (if resume=True)
            model_name: Name of the model (used for new run folder naming)
            config: Configuration dictionary to save
            resume: Whether to resume from existing run_dir
            runs_base_dir: Base directory for runs
        """
        self.resume = resume
        self.runs_base_dir = Path(runs_base_dir)
        
        if resume and run_dir:
            # Resume from existing run
            self.run_dir = Path(run_dir)
            if not self.run_dir.exists():
                raise ValueError(f"Run directory does not exist: {run_dir}")
        else:
            # Create new run directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{model_name}_{timestamp}"
            self.run_dir = self.runs_base_dir / run_name
        
        # Create subdirectories
        self.figs_dir = self.run_dir / "figs"
        self.models_dir = self.run_dir / "models"
        self.probs_dir = self.run_dir / "probs"
        
        self._setup_directories()
        self._setup_logging()
        
        # Save config if provided (and not resuming)
        if config and not resume:
            self.save_config(config)
        
        # CSV writers cache
        self._csv_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, Any] = {}
        self._csv_headers_written: Dict[str, bool] = {}
    
    def _setup_directories(self):
        """Create run directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.probs_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup file and console logging."""
        log_file = self.run_dir / "train.log"
        
        # Remove existing handlers to avoid duplicates
        self.logger = logging.getLogger(f"vesuvius_{self.run_dir.name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # File handler (append if resuming)
        file_mode = 'a' if self.resume else 'w'
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if self.resume:
            self.logger.info("=" * 60)
            self.logger.info("RESUMING TRAINING")
            self.logger.info("=" * 60)
    
    def save_config(self, config: Dict):
        """Save configuration to YAML file."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f"Config saved to {config_path}")
    
    def load_config(self) -> Dict:
        """Load configuration from run directory."""
        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_csv_path(self, name: str) -> Path:
        """Get path for a CSV file."""
        return self.run_dir / f"{name}.csv"
    
    def _init_csv(self, name: str, headers: List[str]):
        """Initialize CSV file with headers if needed."""
        csv_path = self._get_csv_path(name)
        
        # Check if file exists and has content (for resume)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        
        if self.resume and file_exists:
            # Append mode - no new headers
            self._csv_files[name] = open(csv_path, 'a', newline='')
            self._csv_writers[name] = csv.DictWriter(
                self._csv_files[name], 
                fieldnames=headers,
                extrasaction='ignore'
            )
            self._csv_headers_written[name] = True
        else:
            # Write mode - write headers
            self._csv_files[name] = open(csv_path, 'w', newline='')
            self._csv_writers[name] = csv.DictWriter(
                self._csv_files[name],
                fieldnames=headers,
                extrasaction='ignore'
            )
            self._csv_writers[name].writeheader()
            self._csv_files[name].flush()
            self._csv_headers_written[name] = True
    
    def log_metrics(self, name: str, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics to CSV file.
        
        Args:
            name: CSV file name (e.g., 'train_metrics', 'val_metrics')
            epoch: Current epoch number
            metrics: Dictionary of metric names to values
        """
        all_metrics = {'epoch': epoch, **metrics}
        headers = list(all_metrics.keys())
        
        if name not in self._csv_writers:
            self._init_csv(name, headers)
        
        self._csv_writers[name].writerow(all_metrics)
        self._csv_files[name].flush()
    
    def load_metrics_csv(self, name: str) -> List[Dict]:
        """Load metrics from CSV file."""
        csv_path = self._get_csv_path(name)
        if not csv_path.exists():
            return []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def get_last_epoch(self) -> int:
        """Get the last completed epoch from training metrics."""
        metrics = self.load_metrics_csv('train_metrics')
        if not metrics:
            return 0
        return int(metrics[-1].get('epoch', 0))
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level.lower())(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def get_prob_epoch_dir(self, epoch: int) -> Path:
        """Get directory for saving prediction probabilities for an epoch."""
        epoch_dir = self.probs_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        return epoch_dir
    
    def cleanup_old_prob_dirs(self, current_epoch: int, keep_last_n: int = 3):
        """Remove old probability directories to save space."""
        if keep_last_n <= 0:
            return
        
        epoch_dirs = sorted(self.probs_dir.glob("epoch_*"))
        if len(epoch_dirs) <= keep_last_n:
            return
        
        # Remove oldest directories
        dirs_to_remove = epoch_dirs[:-keep_last_n]
        for dir_path in dirs_to_remove:
            shutil.rmtree(dir_path)
            self.debug(f"Cleaned up old prob dir: {dir_path}")
    
    def plot_metrics(
        self,
        train_csv: str = "train_metrics",
        val_csv: str = "val_metrics"
    ):
        """Generate all plots from logged metrics."""
        train_data = self.load_metrics_csv(train_csv)
        val_data = self.load_metrics_csv(val_csv)
        
        if not train_data and not val_data:
            return
        
        # Plot loss
        self._plot_loss(train_data, val_data)
        
        # Plot learning rate
        self._plot_lr(train_data)
        
        # Plot competition metrics
        self._plot_competition_metrics(val_data)
        
        # Plot standard metrics
        self._plot_standard_metrics(val_data)
    
    def _plot_loss(self, train_data: List[Dict], val_data: List[Dict]):
        """Plot training and validation loss."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_data:
            epochs = [int(d['epoch']) for d in train_data]
            train_loss = [float(d.get('loss', 0)) for d in train_data]
            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        
        if val_data:
            epochs = [int(d['epoch']) for d in val_data]
            val_loss = [float(d.get('loss', 0)) for d in val_data if 'loss' in d]
            if val_loss:
                ax.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'loss.png', dpi=150)
        plt.close(fig)
    
    def _plot_lr(self, train_data: List[Dict]):
        """Plot learning rate schedule."""
        if not train_data or 'lr' not in train_data[0]:
            return
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        epochs = [int(d['epoch']) for d in train_data]
        lrs = [float(d.get('lr', 0)) for d in train_data]
        
        ax.plot(epochs, lrs, 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'learning_rate.png', dpi=150)
        plt.close(fig)
    
    def _plot_competition_metrics(self, val_data: List[Dict]):
        """Plot competition metrics (SurfaceDice, VOI, TopoScore)."""
        if not val_data:
            return
        
        metrics_to_plot = ['competition_score', 'surface_dice', 'voi_score', 'topo_score']
        available_metrics = [m for m in metrics_to_plot if m in val_data[0]]
        
        if not available_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = [int(d['epoch']) for d in val_data]
        colors = ['black', 'blue', 'green', 'red']
        
        for metric, color in zip(available_metrics, colors):
            values = [float(d.get(metric, 0)) for d in val_data]
            label = metric.replace('_', ' ').title()
            linewidth = 3 if metric == 'competition_score' else 2
            ax.plot(epochs, values, color=color, label=label, linewidth=linewidth)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Competition Metrics', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'competition_metrics.png', dpi=150)
        plt.close(fig)
    
    def _plot_standard_metrics(self, val_data: List[Dict]):
        """Plot standard segmentation metrics (Dice, IoU)."""
        if not val_data:
            return
        
        metrics_to_plot = ['dice', 'iou', 'precision', 'recall']
        available_metrics = [m for m in metrics_to_plot if m in val_data[0]]
        
        if not available_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = [int(d['epoch']) for d in val_data]
        colors = ['purple', 'orange', 'cyan', 'magenta']
        
        for metric, color in zip(available_metrics, colors):
            values = [float(d.get(metric, 0)) for d in val_data]
            label = metric.replace('_', ' ').title()
            ax.plot(epochs, values, color=color, label=label, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Segmentation Metrics', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'segmentation_metrics.png', dpi=150)
        plt.close(fig)
    
    def close(self):
        """Close all open file handles."""
        for name, f in self._csv_files.items():
            if not f.closed:
                f.close()
        self._csv_files.clear()
        self._csv_writers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
