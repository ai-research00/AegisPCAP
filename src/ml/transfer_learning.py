"""
Transfer Learning Framework for Phase 10: Advanced Analytics

Implements:
- Pre-trained model loading and feature extraction
- Domain adaptation techniques (CORAL, adversarial)
- Fine-tuning strategies with layer-wise learning rates
- Model freezing and gradual unfreezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import torchvision.models as models
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ===== DATA STRUCTURES =====

@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    
    model_name: str  # 'resnet50', 'efficientnet_b0', 'vit_base', etc.
    pretrained: bool = True
    freeze_backbone: bool = True
    freeze_until_layer: Optional[int] = None  # Gradual unfreezing
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    use_layer_wise_lr: bool = True
    layer_wise_lr_decay: float = 0.9  # Each layer gets 90% of prev layer's LR
    batch_size: int = 32
    num_epochs: int = 30
    patience: int = 5
    weight_decay: float = 1e-4


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation."""
    
    method: str = 'coral'  # 'coral', 'adversarial', 'mmd'
    source_weight: float = 1.0
    domain_weight: float = 0.1
    use_batch_norm_adaptation: bool = True
    target_update_frequency: int = 50  # Update batch norm with target data


# ===== FEATURE EXTRACTORS =====

class PretrainedEncoder(nn.Module):
    """
    Wraps pre-trained models for feature extraction.
    
    Supports: ResNet, EfficientNet, Vision Transformers, and custom models.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze: bool = True,
        output_dim: Optional[int] = None
    ):
        """
        Initialize pre-trained encoder.
        
        Args:
            model_name: Name of pre-trained model ('resnet50', 'efficientnet_b0', etc.)
            pretrained: Whether to load pre-trained weights
            freeze: Whether to freeze backbone parameters
            output_dim: Optional dimension for projection head
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze
        self.output_dim = output_dim
        
        # Load pre-trained model
        self.backbone = self._load_backbone(model_name, pretrained)
        
        # Extract feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Optional projection head
        if output_dim and output_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = None
            self.output_dim = self.feature_dim
        
        # Freeze backbone if requested
        if freeze:
            self._freeze_backbone()
    
    def _load_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Load pre-trained backbone."""
        if model_name.startswith('resnet'):
            # ResNet family
            model = getattr(models, model_name)(pretrained=pretrained)
            # Remove classification head
            return nn.Sequential(*list(model.children())[:-1])
        
        elif model_name.startswith('efficientnet'):
            # EfficientNet family
            model = getattr(models, model_name)(pretrained=pretrained)
            return model.features
        
        elif model_name == 'vit_base':
            # Vision Transformer
            model = models.vision_transformer.vit_b_16(pretrained=pretrained)
            return nn.Sequential(
                model.conv_proj,
                nn.Flatten(),
                nn.Linear(model.num_classes, 768)  # Intermediate dimension
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _get_feature_dim(self) -> int:
        """Get output dimension of backbone."""
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        
        if isinstance(features, torch.Tensor):
            return features.view(features.size(0), -1).size(1)
        else:
            raise RuntimeError("Invalid backbone output")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_until_layer(self, layer_idx: int):
        """
        Freeze layers up to specified index (gradual unfreezing).
        
        Args:
            layer_idx: Freeze up to this layer index
        """
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = (i >= layer_idx)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            Features (B, output_dim)
        """
        # Backbone features
        features = self.backbone(x)
        
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Project if needed
        if self.projection is not None:
            features = self.projection(features)
        
        return features


# ===== DOMAIN ADAPTATION =====

class DomainAdaptationBase(ABC):
    """Abstract base for domain adaptation techniques."""
    
    def __init__(self, config: DomainAdaptationConfig):
        self.config = config
    
    @abstractmethod
    def compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute domain adaptation loss."""
        pass


class CORAL(DomainAdaptationBase):
    """
    CORAL (Correlation Alignment) for domain adaptation.
    
    Aligns second-order statistics (covariance) of source and target features.
    """
    
    def compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CORAL loss (covariance alignment).
        
        Args:
            source_features: Source domain features (B, D)
            target_features: Target domain features (B, D)
        
        Returns:
            CORAL loss (scalar)
        """
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Frobenius norm of difference
        coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        
        return coral_loss / (4.0 * source_features.size(0))
    
    @staticmethod
    def _compute_covariance(features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix with centering."""
        batch_size = features.size(0)
        
        # Center features
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # Covariance
        cov = torch.matmul(features_centered.t(), features_centered)
        cov = cov / (batch_size - 1)
        
        return cov


class AdversarialDomainAdaptation(DomainAdaptationBase):
    """
    Adversarial domain adaptation using domain discriminator.
    
    Trains domain discriminator to distinguish source/target, then
    trains feature extractor to fool discriminator.
    """
    
    def __init__(self, config: DomainAdaptationConfig, feature_dim: int):
        super().__init__(config)
        
        # Domain discriminator: Binary classifier (source vs target)
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial domain loss.
        
        Feature extractor loss: fool discriminator (target looks like source)
        
        Args:
            source_features: Source domain features (B, D)
            target_features: Target domain features (B, D)
        
        Returns:
            Adversarial loss (scalar)
        """
        # Discriminator predictions
        source_pred = self.discriminator(source_features)
        target_pred = self.discriminator(target_features)
        
        # Loss: fool discriminator
        # Make target_pred close to 0 (source label)
        adversarial_loss = -torch.mean(torch.log(1 - torch.sigmoid(target_pred) + 1e-8))
        
        return adversarial_loss
    
    def discriminator_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Discriminator tries to distinguish source (1) from target (0).
        
        Args:
            source_features: Source domain features (B, D)
            target_features: Target domain features (B, D)
        
        Returns:
            Discriminator loss (scalar)
        """
        source_pred = self.discriminator(source_features)
        target_pred = self.discriminator(target_features)
        
        # Binary cross-entropy loss
        source_loss = -torch.mean(torch.log(torch.sigmoid(source_pred) + 1e-8))
        target_loss = -torch.mean(torch.log(1 - torch.sigmoid(target_pred) + 1e-8))
        
        return source_loss + target_loss


class MaximumMeanDiscrepancy(DomainAdaptationBase):
    """
    Maximum Mean Discrepancy (MMD) for domain adaptation.
    
    Aligns feature distributions using Gaussian kernel embeddings.
    """
    
    def __init__(
        self,
        config: DomainAdaptationConfig,
        kernel_type: str = 'rbf',
        kernels: Optional[List[float]] = None
    ):
        super().__init__(config)
        self.kernel_type = kernel_type
        self.kernels = kernels or [0.2, 0.5, 0.9, 1.3]
    
    def compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MMD loss.
        
        Args:
            source_features: Source domain features (B, D)
            target_features: Target domain features (B, D)
        
        Returns:
            MMD loss (scalar)
        """
        if self.kernel_type == 'rbf':
            return self._compute_mmd_rbf(source_features, target_features)
        else:
            return self._compute_mmd_linear(source_features, target_features)
    
    def _compute_mmd_rbf(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MMD with RBF kernel."""
        mmd = 0
        
        for scale in self.kernels:
            # Compute pairwise squared distances
            source_dist = torch.cdist(source, source, p=2) ** 2
            target_dist = torch.cdist(target, target, p=2) ** 2
            mixed_dist = torch.cdist(source, target, p=2) ** 2
            
            # RBF kernel
            source_kernel = torch.exp(-source_dist / (2 * scale ** 2))
            target_kernel = torch.exp(-target_dist / (2 * scale ** 2))
            mixed_kernel = torch.exp(-mixed_dist / (2 * scale ** 2))
            
            # MMD computation
            mmd_scale = (
                torch.mean(source_kernel) + torch.mean(target_kernel) -
                2 * torch.mean(mixed_kernel)
            )
            mmd += mmd_scale
        
        return mmd / len(self.kernels)
    
    def _compute_mmd_linear(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MMD with linear kernel."""
        # Mean embeddings
        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)
        
        # Linear MMD
        mmd = torch.norm(source_mean - target_mean, p=2) ** 2
        
        return mmd


# ===== FINE-TUNING =====

class LayerWiseLearningRate:
    """
    Manages layer-wise learning rates for fine-tuning.
    
    Different layers get different learning rates (typically higher for later layers).
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        lr_decay: float = 0.9,
        skip_layers: Optional[List[str]] = None
    ):
        """
        Initialize layer-wise LR scheduler.
        
        Args:
            model: Model to optimize
            base_lr: Base learning rate for final layer
            lr_decay: Decay factor for each earlier layer
            skip_layers: Layer names to skip (e.g., 'bn' for batch norm)
        """
        self.model = model
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.skip_layers = skip_layers or []
    
    def get_param_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups with layer-wise learning rates.
        
        Returns:
            List of param groups for torch.optim.Optimizer
        """
        param_groups = []
        named_params = list(self.model.named_parameters())
        
        # Group by layer
        layers = {}
        for name, param in named_params:
            layer_name = name.split('.')[0]
            if layer_name not in layers:
                layers[layer_name] = []
            layers[layer_name].append((name, param))
        
        # Assign learning rates
        layer_list = list(layers.items())
        for i, (layer_name, params) in enumerate(layer_list):
            # Skip certain layers
            if any(skip in layer_name for skip in self.skip_layers):
                lr = 0  # No gradient update for skipped layers
            else:
                # Reverse order: later layers get higher LR
                layer_idx = len(layer_list) - 1 - i
                lr = self.base_lr * (self.lr_decay ** layer_idx)
            
            param_group = {
                'params': [p for _, p in params],
                'lr': lr,
                'name': layer_name
            }
            param_groups.append(param_group)
        
        return param_groups


class FineTuningTrainer:
    """
    Trainer for fine-tuning pre-trained models.
    
    Features:
    - Warm-up learning rate scheduling
    - Gradual layer unfreezing
    - Layer-wise learning rates
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TransferConfig,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize fine-tuning trainer.
        
        Args:
            model: Model to fine-tune
            config: Fine-tuning configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer with layer-wise LR if needed
        if config.use_layer_wise_lr:
            param_groups = LayerWiseLearningRate(
                model,
                config.learning_rate,
                config.layer_wise_lr_decay
            ).get_param_groups()
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_epochs
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs - self.config.warmup_epochs
        )
        
        # Sequential composition
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )
    
    def train_epoch(
        self,
        train_loader,
        criterion: nn.Module
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Gradual unfreezing
        if self.config.freeze_until_layer is not None:
            unfreeze_layer = min(
                self.epoch // 5,  # Unfreeze every 5 epochs
                self.config.freeze_until_layer
            )
            if hasattr(self.model, 'freeze_until_layer'):
                self.model.freeze_until_layer(unfreeze_layer)
        
        self.epoch += 1
        self.scheduler.step()
        
        return avg_loss
    
    def validate(
        self,
        val_loader,
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
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def should_stop(self, val_loss: float) -> bool:
        """Check early stopping condition."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience


if __name__ == '__main__':
    # Example usage
    config = TransferConfig(
        model_name='resnet50',
        learning_rate=1e-4,
        num_epochs=30
    )
    
    encoder = PretrainedEncoder('resnet50', pretrained=True, output_dim=512)
    print(f"Feature dimension: {encoder.output_dim}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    features = encoder(dummy_input)
    print(f"Output shape: {features.shape}")
    
    # Domain adaptation
    coral = CORAL(DomainAdaptationConfig())
    source_feats = torch.randn(32, 512)
    target_feats = torch.randn(32, 512)
    loss = coral.compute_domain_loss(source_feats, target_feats)
    print(f"CORAL loss: {loss.item():.4f}")
