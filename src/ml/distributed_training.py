"""
Distributed Training Framework for Phase 10: Advanced Analytics

Implements:
- PyTorch Distributed Data Parallel (DDP)
- Ray Tune for hyperparameter optimization
- Ensemble training methods
- Multi-GPU and multi-node training support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ===== CONFIGURATION =====

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    # DDP settings
    backend: str = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    init_method: str = 'env://'
    rank: int = 0
    world_size: int = 1
    
    # Training settings
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    gradient_clip: Optional[float] = 1.0
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_frequency: int = 5  # Save every N epochs
    resume_from: Optional[str] = None
    
    # Synchronization
    sync_bn: bool = True  # Synchronize batch norm across ranks
    find_unused_parameters: bool = False
    
    # Debugging
    log_frequency: int = 10
    profile: bool = False


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    
    ensemble_size: int = 5
    diversity_loss_weight: float = 0.1
    use_different_seeds: bool = True
    use_different_architectures: bool = False
    use_different_datasets: bool = False
    
    # Combination strategy
    combination_method: str = 'voting'  # 'voting', 'averaging', 'weighted'
    voting_weights: Optional[List[float]] = None


@dataclass
class RayTuneConfig:
    """Configuration for Ray Tune hyperparameter optimization."""
    
    num_samples: int = 10
    num_epochs: int = 20
    gpus_per_trial: float = 1.0
    cpus_per_trial: int = 2
    
    # Search space
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64])
    weight_decay_range: Tuple[float, float] = (1e-5, 1e-3)
    
    # Search algorithm
    search_algorithm: str = 'random'  # 'random', 'grid', 'bayes'


# ===== DISTRIBUTED DATA PARALLEL =====

class DDPTrainer:
    """
    Trainer for distributed data parallel training across GPUs.
    
    Features:
    - Automatic rank detection and initialization
    - Synchronized batch normalization
    - Gradient accumulation
    - Checkpointing with rank awareness
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedTrainingConfig,
        device: torch.device
    ):
        """
        Initialize DDP trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on (cuda:rank)
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Get distributed info
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Wrap model with DDP
        if self.world_size > 1:
            self.model = DDP(
                model.to(device),
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=config.find_unused_parameters
            )
        else:
            self.model = model.to(device)
        
        # Synchronize batch norm if requested
        if config.sync_bn and self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
    
    def _log(self, message: str):
        """Log only from rank 0."""
        if self.rank == 0:
            logger.info(message)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        log_frequency: int = 10
    ) -> float:
        """
        Train for one epoch with gradient accumulation.
        
        Args:
            train_loader: Training data loader (with DistributedSampler)
            criterion: Loss function
            log_frequency: Log every N steps
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        
        # Set sampler epoch for shuffling
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0
        accumulated_loss = 0
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            accumulated_loss += loss.item()
            
            # Accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Synchronize gradients across ranks
                if self.world_size > 1:
                    dist.all_reduce(loss)
                    loss = loss / self.world_size
                
                total_loss += accumulated_loss
                accumulated_loss = 0
                
                # Logging
                if (step + 1) % (log_frequency * self.config.gradient_accumulation_steps) == 0:
                    avg_loss = total_loss / ((step + 1) // self.config.gradient_accumulation_steps)
                    self._log(
                        f"Epoch {self.epoch}, Step {step+1}, Loss: {avg_loss:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
            
            self.global_step += 1
        
        avg_epoch_loss = total_loss / max(
            1,
            (len(train_loader) // self.config.gradient_accumulation_steps)
        )
        
        self.epoch += 1
        self.scheduler.step()
        
        return avg_epoch_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Accuracy
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        # All-reduce metrics across ranks
        if self.world_size > 1:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            correct_tensor = torch.tensor([correct], device=self.device)
            total_tensor = torch.tensor([total], device=self.device)
            
            dist.all_reduce(total_loss_tensor)
            dist.all_reduce(correct_tensor)
            dist.all_reduce(total_tensor)
            
            total_loss = total_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, path: str):
        """Save checkpoint from rank 0 only."""
        if self.rank == 0:
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self._get_model_state(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step': self.global_step
            }
            torch.save(checkpoint, path)
            self._log(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint on all ranks."""
        if not os.path.exists(path):
            self._log(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self._log(f"Checkpoint loaded from {path} (epoch {self.epoch})")
    
    def _get_model_state(self) -> Dict:
        """Get model state dict, handling DDP wrapper."""
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


# ===== ENSEMBLE TRAINING =====

class EnsembleTrainer:
    """
    Trains an ensemble of models with diversity losses.
    
    Techniques:
    - Different random seeds
    - Different architectures
    - Different subsets of data
    - Diversity-promoting loss terms
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: EnsembleConfig,
        device: torch.device
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            model_factory: Callable that returns new model instances
            config: Ensemble configuration
            device: Device to train on
        """
        self.model_factory = model_factory
        self.config = config
        self.device = device
        
        # Create ensemble
        self.models = [
            model_factory().to(device)
            for _ in range(config.ensemble_size)
        ]
        
        # Optimizers
        self.optimizers = [
            torch.optim.AdamW(model.parameters())
            for model in self.models
        ]
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[List[float], float]:
        """
        Train ensemble for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
        
        Returns:
            Tuple of (list of individual losses, diversity loss)
        """
        for model in self.models:
            model.train()
        
        individual_losses = [0] * self.config.ensemble_size
        diversity_loss_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Train each model
            outputs_list = []
            losses = []
            
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Add diversity loss
                if self.config.diversity_loss_weight > 0 and len(outputs_list) > 0:
                    diversity_term = self._compute_diversity_loss(
                        outputs,
                        outputs_list[-1]  # Compare with previous model
                    )
                    loss = loss + self.config.diversity_loss_weight * diversity_term
                
                loss.backward()
                optimizer.step()
                
                individual_losses[i] += loss.item()
                losses.append(loss.item())
                outputs_list.append(outputs.detach())
            
            # Diversity loss across all models
            if len(outputs_list) > 1:
                diversity_loss = self._compute_pairwise_diversity(outputs_list)
                diversity_loss_total += diversity_loss
        
        avg_losses = [
            loss / len(train_loader)
            for loss in individual_losses
        ]
        avg_diversity = diversity_loss_total / len(train_loader)
        
        return avg_losses, avg_diversity
    
    @staticmethod
    def _compute_diversity_loss(outputs1: torch.Tensor, outputs2: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss between two models.
        
        Encourages different predictions (high variance).
        """
        # KL divergence between softmax outputs
        probs1 = torch.softmax(outputs1, dim=1)
        probs2 = torch.softmax(outputs2, dim=1)
        
        # Symmetric KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.log(probs1 + 1e-8),
            probs2,
            reduction='batchmean'
        )
        
        return -kl_div  # Negative to maximize divergence
    
    @staticmethod
    def _compute_pairwise_diversity(outputs_list: List[torch.Tensor]) -> float:
        """Compute average pairwise diversity across ensemble."""
        diversity = 0
        count = 0
        
        for i in range(len(outputs_list)):
            for j in range(i + 1, len(outputs_list)):
                diversity += EnsembleTrainer._compute_diversity_loss(
                    outputs_list[i],
                    outputs_list[j]
                ).item()
                count += 1
        
        return diversity / count if count > 0 else 0
    
    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Tuple of (ensemble predictions, standard deviation)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (ensemble_size, batch_size, num_classes)
        
        # Average prediction
        ensemble_pred = torch.mean(predictions, dim=0)
        
        # Uncertainty (standard deviation)
        uncertainty = torch.std(predictions, dim=0)
        
        return ensemble_pred, uncertainty


# ===== RAY TUNE INTEGRATION =====

class RayTuneTrainer:
    """
    Hyperparameter optimization with Ray Tune.
    
    Supports:
    - Random search
    - Grid search
    - Bayesian optimization
    - Parallel trials
    """
    
    def __init__(
        self,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        config: RayTuneConfig
    ):
        """
        Initialize Ray Tune trainer.
        
        Args:
            model_factory: Callable that creates model given hyperparams
            config: Ray Tune configuration
        """
        self.model_factory = model_factory
        self.config = config
        self.best_config = None
        self.best_loss = float('inf')
    
    def get_search_space(self) -> Dict[str, Any]:
        """Get hyperparameter search space for Ray Tune."""
        from ray import tune
        
        return {
            'learning_rate': tune.loguniform(
                self.config.learning_rate_range[0],
                self.config.learning_rate_range[1]
            ),
            'batch_size': tune.choice(self.config.batch_size_options),
            'weight_decay': tune.loguniform(
                self.config.weight_decay_range[0],
                self.config.weight_decay_range[1]
            )
        }
    
    def train_trial(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        num_epochs: int
    ) -> Dict[str, float]:
        """
        Train a single trial with given hyperparameters.
        
        Args:
            config: Hyperparameter configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            num_epochs: Number of epochs to train
        
        Returns:
            Metrics dict for Ray Tune
        """
        from ray import train
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = self.model_factory(config).to(device)
        
        # Create optimizer with config hyperparams
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
            
            val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            # Report metrics
            train.report({
                'loss': val_loss,
                'accuracy': accuracy
            })
            
            # Update best config
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_config = config
        
        return {'loss': val_loss, 'accuracy': accuracy}


def init_distributed_training():
    """
    Initialize distributed training environment.
    
    Must be called before creating models.
    """
    if not dist.is_available():
        logger.warning("Torch distributed is not available")
        return
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://'
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")


def cleanup_distributed_training():
    """Clean up distributed training environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
