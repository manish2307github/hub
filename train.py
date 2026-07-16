"""
Training script for AGHNN
Implements two-stage training with green regularization and knowledge distillation
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import config
from torch.serialization import add_safe_globals
from config import ExperimentConfig

from dataclasses import asdict

from models import AGHNN_Tiny, AGHNN_Small, AGHNN_Base, AGHNN_Large
from utils.data import get_dataloaders, CutMix, MixUp
from utils.metrics import AverageMeter, accuracy, compute_complexity_statistics
from utils.logger import setup_logger, log_metrics, TensorBoardLogger, ExperimentTracker
from utils.energy import EnergyProfiler, compute_flops, compute_parameters
from config import ExperimentConfig, get_cifar10_config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(config: ExperimentConfig) -> nn.Module:
    """Create model based on configuration"""
    model_variants = {
        'tiny': AGHNN_Tiny,
        'small': AGHNN_Small,
        'base': AGHNN_Base,
        'large': AGHNN_Large
    }
    
    model_fn = model_variants.get(config.model.variant.lower(), AGHNN_Small)
    return model_fn(num_classes=config.model.num_classes)


def get_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    """Create optimizer"""
    return optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        nesterov=True
    )


def get_scheduler(
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
    steps_per_epoch: int
) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    total_steps = config.training.epochs * steps_per_epoch
    warmup_steps = config.training.warmup_epochs * steps_per_epoch
    
    if config.training.lr_schedule == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    elif config.training.lr_schedule == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30 * steps_per_epoch, gamma=0.1
        )
    else:
        main_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[60 * steps_per_epoch, 120 * steps_per_epoch, 160 * steps_per_epoch],
            gamma=0.2
        )
    
    # Warmup scheduler
    if config.training.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = main_scheduler
    
    return scheduler


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: ExperimentConfig,
    epoch: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined loss with green regularization and knowledge distillation
    
    Args:
        outputs: Model outputs dictionary
        targets: Ground truth labels
        config: Experiment configuration
        epoch: Current epoch (for scheduling)
        
    Returns:
        Total loss and dictionary of individual losses
    """
    logits = outputs['logits']
    easy_logits = outputs['easy_logits']
    hard_logits = outputs['hard_logits']
    
    # Task loss with label smoothing
    if config.training.label_smoothing > 0:
        ce_loss = F.cross_entropy(
            logits, targets, 
            label_smoothing=config.training.label_smoothing
        )
    else:
        ce_loss = F.cross_entropy(logits, targets)
    
    total_loss = ce_loss
    loss_dict = {'ce_loss': ce_loss.item()}
    
    # ✅ Complexity regularization loss (CRITICAL: prevents collapse)
    # Force complexity to learn bimodal distribution (peaks at 0 and 1)
    if 'complexity' in outputs:
        complexity = outputs['complexity']
        # Push toward extremes (0=easy, 1=hard) instead of middle
        complexity_reg_loss = (complexity ** 2).mean() + ((1 - complexity) ** 2).mean()
        total_loss = total_loss + 0.05 * complexity_reg_loss
        loss_dict['complexity_reg_loss'] = complexity_reg_loss.item()
    
    # Energy loss (green regularization)
        # Energy loss (green regularization) - deferred until routing learns
    if config.training.lambda_energy > 0 and 'complexity' in outputs:
        complexity = outputs['complexity']
        
        # Start energy loss only after easy path has trained independently
        if epoch >= getattr(config.training, 'lambda_energy_start_epoch', 0):
            if config.training.lambda_energy_schedule == 'linear':
                # Scale linearly from start_epoch onwards
                progress = (epoch - config.training.lambda_energy_start_epoch) / (
                    config.training.epochs - config.training.lambda_energy_start_epoch
                )
                lambda_e = (
                    config.training.lambda_energy_start + 
                    (config.training.lambda_energy_end - config.training.lambda_energy_start) * progress
                )
            else:
                lambda_e = config.training.lambda_energy
        else:
            lambda_e = 0.0  # No energy penalty before start_epoch

        # Energy loss: penalize using hard path
        energy_loss = complexity.mean()
        total_loss = total_loss + lambda_e * energy_loss
        loss_dict['energy_loss'] = energy_loss.item()
        loss_dict['lambda_energy'] = lambda_e
    
    # Knowledge distillation loss
        # Knowledge distillation loss - only after easy path has basic competence
    # Apply only after epoch 50 to avoid forcing premature alignment
    if config.training.use_distillation and epoch >= 50:
        T = config.training.distill_temperature
        alpha = config.training.distill_alpha
        
        # Hard path teaches easy path
        soft_targets = F.softmax(hard_logits / T, dim=1)
        soft_predictions = F.log_softmax(easy_logits / T, dim=1)
        
        distill_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (T ** 2)
        total_loss = total_loss + alpha * distill_loss
        loss_dict['distill_loss'] = distill_loss.item()
        
    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: ExperimentConfig,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    cutmix: Optional[CutMix] = None,
    mixup: Optional[MixUp] = None
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary of average metrics
    """
    model.train()
    device = next(model.parameters()).device
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc@1')
    complexity_meter = AverageMeter('Complexity')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        # Apply CutMix or MixUp
        mixed_targets = None
        lam = 1.0
        
        if cutmix and random.random() < config.training.cutmix_prob:
            images, targets, mixed_targets, lam = cutmix(images, targets)
        elif mixup and random.random() < config.training.mixup_prob:
            images, targets, mixed_targets, lam = mixup(images, targets)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            outputs = model(images, return_complexity=True)
            
            if mixed_targets is not None:
                # Mixed loss for CutMix/MixUp
                loss1, _ = compute_loss(outputs, targets, config, epoch)
                loss2, _ = compute_loss(outputs, mixed_targets, config, epoch)
                loss = lam * loss1 + (1 - lam) * loss2
                _, loss_dict = compute_loss(outputs, targets, config, epoch)
            else:
                loss, loss_dict = compute_loss(outputs, targets, config, epoch)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Metrics
        acc1 = accuracy(outputs['logits'], targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        
        if 'complexity' in outputs:
            complexity_meter.update(outputs['complexity'].mean().item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return {
        'loss': loss_meter.avg,
        'acc1': acc_meter.avg,
        'complexity': complexity_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    hard_routing: bool = False
) -> Dict[str, float]:
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        config: Experiment configuration
        hard_routing: Whether to use hard routing
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    loss_meter = AverageMeter('Loss')
    acc1_meter = AverageMeter('Acc@1')
    acc5_meter = AverageMeter('Acc@5')
    complexity_scores = []
    
    for images, targets in tqdm(val_loader, desc='Validating'):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images, return_complexity=True, hard_routing=hard_routing)
        
        loss = F.cross_entropy(outputs['logits'], targets)
        acc1, acc5 = accuracy(outputs['logits'], targets, topk=(1, 5))
        
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
        
        if 'complexity' in outputs:
            complexity_scores.append(outputs['complexity'].cpu())
    
    metrics = {
        'loss': loss_meter.avg,
        'acc1': acc1_meter.avg,
        'acc5': acc5_meter.avg
    }
    
    if complexity_scores:
        all_complexity = torch.cat(complexity_scores)
        complexity_stats = compute_complexity_statistics(
            all_complexity,
            config.model.threshold_easy,
            config.model.threshold_hard
        )
        metrics.update({
            'complexity_mean': complexity_stats['mean'],
            'easy_ratio': complexity_stats['easy_ratio'],
            'hard_ratio': complexity_stats['hard_ratio']
        })
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: ExperimentConfig,
    is_best: bool = False,
    best_acc: float = 0.0
):
    """Save model checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'best_acc': best_acc,
        'config': config
    }
    
    # Save latest
    torch.save(checkpoint, os.path.join(config.save_dir, 'latest.pth'))
    
    # Save periodic
    if epoch % config.save_interval == 0:
        torch.save(checkpoint, os.path.join(config.save_dir, f'epoch_{epoch}.pth'))
    
    # Save best
    if is_best:
        torch.save(checkpoint, os.path.join(config.save_dir, 'best.pth'))


def _load_best_acc_from_best_checkpoint(save_dir: str) -> float:
    """Load historical best accuracy from best checkpoint if available."""
    best_path = os.path.join(save_dir, 'best.pth')
    if not os.path.exists(best_path):
        return 0.0

    try:

        torch.serialization.add_safe_globals([config.TrainingConfig])
        checkpoint = torch.load(best_path, map_location=device,  weights_only=False)

        if isinstance(best_checkpoint, dict):
            if 'best_acc' in best_checkpoint:
                return float(best_checkpoint['best_acc'])
            if isinstance(best_checkpoint.get('metrics'), dict):
                return float(best_checkpoint['metrics'].get('acc1', 0.0))
    except Exception:
        return 0.0

    return 0.0


def resume_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    resume_path: str,
    device: torch.device,
    logger
) -> Tuple[int, float]:
    """Resume model, optimizer, and scheduler states from checkpoint.

    Returns:
        start_epoch: Next epoch to run
        best_acc: Best validation accuracy so far
    """

    #add_safe_globals([ExperimentConfig])
    torch.serialization.add_safe_globals([config.TrainingConfig])

    checkpoint = torch.load(resume_path, map_location=device,  weights_only=False)

    
    # with safe_globals([ExperimentConfig]):
    #     checkpoint = torch.load(
    #         resume_path,
    #         map_location=device,
    #         weights_only=False
    #     )


    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        last_epoch = int(checkpoint.get('epoch', 0))

        if 'best_acc' in checkpoint:
            best_acc = float(checkpoint['best_acc'])
        elif isinstance(checkpoint.get('metrics'), dict):
            best_acc = float(checkpoint['metrics'].get('acc1', 0.0))
        else:
            best_acc = 0.0

        logger.info(f"Resumed from checkpoint: {resume_path}")
        logger.info(f"Last completed epoch: {last_epoch}")
        logger.info(f"Best validation accuracy so far: {best_acc:.2f}%")
        return last_epoch + 1, best_acc

    # Support raw state_dict checkpoints for partial resume.
    model.load_state_dict(checkpoint)
    logger.warning("Resume checkpoint does not contain optimizer/scheduler states.")
    logger.warning("Resuming with model weights only from epoch 1.")
    return 1, 0.0


def train(config: ExperimentConfig, resume_path: Optional[str] = None):
    """
    Main training function
    
    Args:
        config: Experiment configuration
    """
    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Logging
    logger = setup_logger('train', config.log_dir)
    tb_logger = TensorBoardLogger(os.path.join(config.log_dir, 'tensorboard'))
    tracker = ExperimentTracker(config.experiment_name, config.log_dir)
    tracker.set_config(asdict(config))
    
    logger.info(f"Training AGHNN on {config.data.dataset}")
    logger.info(f"Device: {device}")
    
    # Data
    train_loader, val_loader, dataset_info = get_dataloaders(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        img_size=config.data.img_size
    )
    
    logger.info(f"Dataset: {dataset_info}")
    
    # Model
    model = get_model(config).to(device)
    
    # Log model info
    param_info = compute_parameters(model)
    logger.info(f"Model parameters: {param_info['total_formatted']}")
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Data augmentation
    cutmix = CutMix(config.training.cutmix_alpha, config.training.cutmix_prob) if config.training.use_cutmix else None
    mixup = MixUp(config.training.mixup_alpha, config.training.mixup_prob) if config.training.use_mixup else None
    
    # Training loop
    best_acc = _load_best_acc_from_best_checkpoint(config.save_dir)
    start_epoch = 1

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, resumed_best_acc = resume_training_state(
            model, optimizer, scheduler, resume_path, device, logger
        )
        best_acc = max(best_acc, resumed_best_acc)
    
    if start_epoch > config.training.epochs:
        logger.info(
            f"Checkpoint already at epoch {start_epoch - 1}, which is >= target "
            f"epochs {config.training.epochs}. Nothing to train."
        )
        return model, best_acc

    for epoch in range(start_epoch, config.training.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, scaler, cutmix, mixup
        )
        
        # Validate
        val_metrics = validate(model, val_loader, config, hard_routing=config.hard_routing)
        
        # Logging
        log_metrics(logger, epoch, train_metrics, 'train')
        log_metrics(logger, epoch, val_metrics, 'val')
        
        tb_logger.log_scalars('loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
        tb_logger.log_scalars('accuracy', {'train': train_metrics['acc1'], 'val': val_metrics['acc1']}, epoch)
        
        tracker.log_epoch(epoch, train_metrics, val_metrics)
        
        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc
        if is_best:
            best_acc = val_metrics['acc1']
            
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, config, is_best, best_acc
        )
        
        logger.info(f"Epoch {epoch}: Val Acc = {val_metrics['acc1']:.2f}% (Best: {best_acc:.2f}%)")
    
    # Final evaluation
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    
    tracker.set_best_metrics({'acc1': best_acc})
    tracker.set_final_metrics(val_metrics)
    tracker.save()
    
    tb_logger.close()
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train AGHNN')
    parser.add_argument('--config', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Configuration preset')
    parser.add_argument('--model', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model variant')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--save-interval', type=int, default=None,
                       help='Checkpoint save interval in epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help="Checkpoint path to resume from or 'latest' to use save-dir/latest.pth")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'cifar10':
        from config import get_cifar10_config
        config = get_cifar10_config()
    elif args.config == 'cifar100':
        from config import get_cifar100_config
        config = get_cifar100_config()
    else:
        from config import get_imagenet_config
        config = get_imagenet_config()
    
    # Override with command line arguments
    config.model.variant = args.model
    config.seed = args.seed
    config.device = args.device
    config.data.data_dir = args.data_dir
    config.save_dir = args.save_dir
    
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.save_interval:
        config.save_interval = args.save_interval

    resume_path = os.path.join(config.save_dir, 'latest.pth')
    if not os.path.exists(resume_path):
        resume_path = None
    
    # Train
    train(config, resume_path=resume_path)


if __name__ == '__main__':
    main()
