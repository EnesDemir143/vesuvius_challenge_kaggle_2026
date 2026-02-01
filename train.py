#!/usr/bin/env python
"""
Training script for Vesuvius Challenge Surface Detection.

Usage:
    # New training
    python train.py --model segresnet --epochs 100 --batch_size 2
    
    # Resume training
    python train.py --resume runs/segresnet_2024-02-01_12-00-00
    
    # Override config options
    python train.py --lr 1e-3 --epochs 50 --model unet
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import VesuviusLMDBDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.training import ExperimentLogger, Trainer
from src.utils.config import get_config


def get_model(model_name: str, in_channels: int = 1, out_channels: int = 2) -> nn.Module:
    """
    Get model by name.
    
    Args:
        model_name: Model name ('unet', 'segresnet', 'segformer3d')
        in_channels: Number of input channels
        out_channels: Number of output classes
        
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == "unet":
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_name == "segresnet":
        from monai.networks.nets import SegResNet
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif model_name == "segformer3d" or model_name == "segformer":
        try:
            # Try segmentation_models_pytorch_3d first
            import segmentation_models_pytorch_3d as smp3d
            model = smp3d.Unet(
                encoder_name="mit_b2",  # MixVisionTransformer (SegFormer backbone)
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
            )
        except ImportError:
            # Fallback to MONAI SwinUNETR (transformer-based alternative)
            from monai.networks.nets import SwinUNETR
            model = SwinUNETR(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_v2=True,
            )
            print("Note: Using SwinUNETR as SegFormer3D fallback. Install segmentation_models_pytorch_3d for native SegFormer.")
    elif model_name == "swinunetr" or model_name == "swin":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_v2=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Options: unet, segresnet, segformer3d, swinunetr")
    
    return model


def get_layer_groups(model: nn.Module, model_name: str) -> list:
    """
    Get layer groups for layer-wise learning rate decay.
    
    Returns list of parameter groups from deepest (encoder) to shallowest (decoder).
    """
    model_name = model_name.lower()
    
    if model_name == "unet":
        # MONAI UNet: model.down, model.up
        groups = []
        if hasattr(model, 'model'):
            # Wrapped model
            m = model.model
        else:
            m = model
        
        # Encoder blocks (deepest layers)
        encoder_params = []
        decoder_params = []
        for name, param in m.named_parameters():
            if 'down' in name or 'conv_0' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Split encoder into halves
        mid = len(encoder_params) // 2
        groups = [
            encoder_params[:mid],      # Deep encoder (lowest LR)
            encoder_params[mid:],      # Shallow encoder
            decoder_params,            # Decoder (highest LR)
        ]
        return groups
    
    elif model_name == "segresnet":
        # MONAI SegResNet: down_layers, up_layers
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'down_layers' in name or 'conv_init' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        mid = len(encoder_params) // 2
        groups = [
            encoder_params[:mid],
            encoder_params[mid:],
            decoder_params,
        ]
        return groups
    
    elif model_name in ["segformer3d", "segformer", "swinunetr", "swin"]:
        # Transformer models: encoder stages + decoder
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if any(x in name for x in ['encoder', 'patch_embed', 'stages', 'swinViT', 'layers']):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Split encoder into 3 groups (4 stages typically)
        n = len(encoder_params)
        g1 = n // 3
        g2 = 2 * n // 3
        groups = [
            encoder_params[:g1],       # Deep encoder (lowest LR)
            encoder_params[g1:g2],     # Middle encoder
            encoder_params[g2:],       # Shallow encoder
            decoder_params,            # Decoder (highest LR)
        ]
        return groups
    
    else:
        # Fallback: single group
        return [list(model.parameters())]


def setup_finetune(
    model: nn.Module, 
    model_name: str, 
    config: dict
) -> tuple:
    """
    Setup fine-tuning with layer-wise LR and optional freezing.
    
    Args:
        model: PyTorch model
        model_name: Model name
        config: Training config (merged with model-specific)
        
    Returns:
        (param_groups, num_frozen_params)
    """
    finetune_mode = config.get('finetune_mode', 'full')
    base_lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    lr_decay_rate = config.get('lr_decay_rate', 0.75)
    freeze_ratio = config.get('freeze_encoder_ratio', 0.5)
    
    layer_groups = get_layer_groups(model, model_name)
    n_groups = len(layer_groups)
    
    param_groups = []
    num_frozen = 0
    
    if finetune_mode == "head_only":
        # Freeze all encoder, train only decoder (last group)
        for i, group in enumerate(layer_groups[:-1]):
            for param in group:
                param.requires_grad = False
                num_frozen += param.numel()
        
        # Only decoder trains
        param_groups.append({
            'params': layer_groups[-1],
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'decoder'
        })
    
    elif finetune_mode == "middle":
        # Freeze early encoder layers, layer-wise LR for rest
        freeze_groups = int(n_groups * freeze_ratio)
        
        for i, group in enumerate(layer_groups):
            if i < freeze_groups:
                # Freeze this group
                for param in group:
                    param.requires_grad = False
                    num_frozen += param.numel()
            else:
                # Train with layer-wise LR decay
                # Deeper layers get lower LR
                depth_from_decoder = n_groups - 1 - i
                lr_mult = lr_decay_rate ** depth_from_decoder
                group_lr = base_lr * lr_mult
                
                param_groups.append({
                    'params': group,
                    'lr': group_lr,
                    'weight_decay': weight_decay,
                    'name': f'layer_group_{i}'
                })
    
    else:  # full finetune
        # Layer-wise LR decay without freezing
        for i, group in enumerate(layer_groups):
            depth_from_decoder = n_groups - 1 - i
            lr_mult = lr_decay_rate ** depth_from_decoder
            group_lr = base_lr * lr_mult
            
            param_groups.append({
                'params': group,
                'lr': group_lr,
                'weight_decay': weight_decay,
                'name': f'layer_group_{i}'
            })
    
    return param_groups, num_frozen


def get_optimizer(
    model: nn.Module, 
    config: dict, 
    model_name: str = "segresnet"
) -> tuple:
    """
    Get optimizer with layer-wise LR and freezing support.
    
    Returns:
        (optimizer, num_frozen_params)
    """
    name = config.get('optimizer', 'adamw').lower()
    
    # Setup fine-tuning param groups
    param_groups, num_frozen = setup_finetune(model, model_name, config)
    
    if not param_groups:
        # Fallback if no trainable params
        param_groups = [{'params': model.parameters(), 'lr': config.get('learning_rate', 1e-4)}]
    
    if name == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer, num_frozen


def get_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: dict, 
    steps_per_epoch: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler based on config."""
    name = config.get('scheduler', 'cosine').lower()
    epochs = config.get('epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 5)
    
    if name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif name == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    elif name == 'none' or name is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def get_loss_function(config: dict) -> nn.Module:
    """Get loss function."""
    from monai.losses import DiceCELoss
    return DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
    )


def load_full_config(config_path: str = "config.yaml", resume_from: Optional[str] = None) -> dict:
    """
    Load full configuration.
    
    Args:
        config_path: Path to main config file
        resume_from: Path to run folder to resume from
        
    Returns:
        Full configuration dictionary
    """
    if resume_from:
        # Load config from existing run
        run_config_path = Path(resume_from) / "config.yaml"
        if run_config_path.exists():
            with open(run_config_path, 'r') as f:
                return yaml.safe_load(f)
    
    # Load from main config
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Vesuvius Surface Detection Model")
    
    # Model
    parser.add_argument('--model', type=str, default=None, 
                        help='Model name (unet, segresnet, segformer3d, swinunetr)')
    
    # Tune mode
    parser.add_argument('--tune', type=str, default='middle',
                        choices=['linear_probe', 'middle', 'full'],
                        help='Fine-tuning mode: linear_probe (head only), middle (freeze encoder), full')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Run folder to resume from')
    
    # Config
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    # Device
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, cpu)')
    
    args = parser.parse_args()
    
    # Load base configuration
    config = load_full_config(args.config, args.resume)
    train_config = config.get('training', {})
    data_config = config.get('data', {})
    aug_config = config.get('augmentation', {})
    
    # Get model name
    model_name = args.model or train_config.get('model', 'segresnet')
    train_config['model'] = model_name
    
    # Get tune mode settings from model-specific config
    models_config = config.get('models', {})
    model_settings = models_config.get(model_name, {})
    tune_settings = model_settings.get(args.tune, {})
    
    # Merge: base training -> model tune settings -> CLI overrides
    merged_config = {**train_config, **tune_settings}
    
    # Override with CLI args
    if args.epochs:
        merged_config['epochs'] = args.epochs
    if args.batch_size:
        merged_config['batch_size'] = args.batch_size
    if args.lr:
        merged_config['learning_rate'] = args.lr
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resume handling
    resume_from = args.resume or train_config.get('resume_from')
    is_resume = resume_from is not None
    
    # Setup logger
    model_name = train_config.get('model', 'segresnet')
    logger = ExperimentLogger(
        run_dir=resume_from if is_resume else None,
        model_name=model_name,
        config=config if not is_resume else None,
        resume=is_resume
    )
    
    logger.info(f"Configuration: {train_config}")
    logger.info(f"Device: {device}")
    
    # Data loaders
    train_lmdb = data_config.get('train_lmdb', 'dataset/processed/train.lmdb')
    stats_path = aug_config.get('stats_path', 'dataset/processed/mean_std.npy')
    z_jitter = aug_config.get('z_jitter_range', 5)
    
    train_transform = get_train_transforms(stats_path=stats_path, z_jitter_range=z_jitter)
    val_transform = get_val_transforms(stats_path=stats_path)
    
    train_dataset = VesuviusLMDBDataset(
        lmdb_path=train_lmdb,
        transform=train_transform,
        split='train',
        val_ratio=data_config.get('val_ratio', 0.2),
        seed=data_config.get('seed', 42)
    )
    
    val_dataset = VesuviusLMDBDataset(
        lmdb_path=train_lmdb,
        transform=val_transform,
        split='val',
        val_ratio=data_config.get('val_ratio', 0.2),
        seed=data_config.get('seed', 42)
    )
    
    batch_size = train_config.get('batch_size', 2)
    # Optimized DataLoader settings for RAM efficiency
    num_workers = data_config.get('num_workers', 2)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,  # Reuse workers to avoid respawn overhead
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Tune mode: {args.tune}")
    logger.info(f"Finetune mode: {merged_config.get('finetune_mode', 'full')}")
    logger.info(f"Learning rate: {merged_config.get('learning_rate', 1e-4)}")
    logger.info(f"LR decay rate: {merged_config.get('lr_decay_rate', 0.75)}")
    
    # Model
    model = get_model(
        model_name,
        in_channels=merged_config.get('in_channels', 1),
        out_channels=merged_config.get('out_channels', 2)
    )
    model = model.to(device)
    
    # Optimizer with layer-wise LR and freezing
    optimizer, num_frozen = get_optimizer(model, merged_config, model_name)
    
    # Log training info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Frozen parameters: {num_frozen:,} ({100*num_frozen/total_params:.1f}%)")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log per-group learning rates
    for i, pg in enumerate(optimizer.param_groups):
        logger.info(f"  Group {i} ({pg.get('name', 'unnamed')}): LR={pg['lr']:.2e}, params={sum(p.numel() for p in pg['params']):,}")
    
    # Scheduler
    scheduler = get_scheduler(optimizer, merged_config, len(train_loader))
    
    # Loss function
    loss_fn = get_loss_function(merged_config)
    
    # Calculate prob saving interval (auto mode: epochs/50, min 1)
    epochs = train_config.get('epochs', 100)
    prob_interval = train_config.get('save_probs_every_n_epochs', 'auto')
    if prob_interval == 'auto' or prob_interval is None:
        # Golden ratio: for 100 epochs -> 2, for 50 -> 1, for 200 -> 4
        prob_interval = max(1, epochs // 50)
    else:
        prob_interval = int(prob_interval)
    
    logger.info(f"Prob saving interval: every {prob_interval} epochs")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        logger=logger,
        device=device,
        use_amp=train_config.get('use_amp', True),
        grad_accum_steps=train_config.get('grad_accum_steps', 1),
        clip_grad_norm=train_config.get('clip_grad_norm', 1.0),
        save_every_n_epochs=train_config.get('save_every_n_epochs', 5),
        save_probs_every_n_epochs=prob_interval,
        keep_last_n_prob_epochs=train_config.get('keep_last_n_prob_epochs', 3),
        early_stopping_patience=train_config.get('early_stopping_patience', 20),
        early_stopping_metric=train_config.get('early_stopping_metric', 'competition_score'),
        metrics_tau=train_config.get('surface_dice_tau', 2.0),
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    if is_resume:
        start_epoch = trainer.load_checkpoint(resume_from)
    
    # Train
    epochs = train_config.get('epochs', 100)
    result = trainer.train(epochs=epochs, start_epoch=start_epoch)
    
    logger.info(f"Training completed! Best {train_config.get('early_stopping_metric', 'competition_score')}: {result['best_metric']:.4f}")
    logger.close()


if __name__ == "__main__":
    main()
