"""
Training module for Vesuvius Challenge Surface Detection.

Provides:
- ExperimentLogger: Run folder management, CSV logging, plotting
- Trainer: Training loop with checkpointing and resume support
"""

from .experiment_logger import ExperimentLogger
from .trainer import Trainer

__all__ = [
    "ExperimentLogger",
    "Trainer",
]
