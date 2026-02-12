"""Knowledge distillation for model compression.

Trains a smaller student model to mimic a larger teacher model.
Preserves teacher accuracy while reducing model size and inference latency.

Classes:
    KnowledgeDistiller: Teacher-student training with distillation loss
    DistillationLoss: Kullback-Leibler divergence-based loss
    TemperatureScaling: Output temperature adjustment
"""

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class DistillationLoss(nn.Module):
    """Kullback-Leibler divergence-based distillation loss.
    
    Combines task loss (cross-entropy) with distillation loss (KL divergence).
    
    Attributes:
        temperature: Temperature for softening probabilities (default: 4.0)
        alpha: Weight for distillation loss (default: 0.5)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening (higher = softer)
            alpha: Weight for distillation loss relative to task loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model output (batch_size, num_classes)
            teacher_logits: Teacher model output (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Combined loss value
        """
        # Task loss (cross-entropy with ground truth)
        task_loss = self.task_loss(student_logits, target)
        
        # Distillation loss (KL divergence between softened probabilities)
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean',
        )
        
        # Combined loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return total_loss


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration.
    
    Adjusts model output logits with a learned temperature parameter
    to improve probability calibration.
    
    Attributes:
        temperature: Temperature parameter (default: 1.0 = no scaling)
    """

    def __init__(self, initial_temp: float = 1.0):
        """Initialize temperature scaling.
        
        Args:
            initial_temp: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(initial_temp, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits with learned temperature.
        
        Args:
            logits: Model output logits
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature

    def get_temperature(self) -> float:
        """Get current temperature value.
        
        Returns:
            Temperature value
        """
        return self.temperature.item()


class KnowledgeDistiller:
    """Knowledge distillation trainer.
    
    Trains a student model to mimic a teacher model using distillation loss.
    Teacher model should be pre-trained; student model is trained from scratch.
    
    Attributes:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Student model optimizer
        device: Device to train on
        temperature: Distillation temperature
        alpha: Weight for distillation loss
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        device: str = "cpu",
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """Initialize knowledge distiller.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for student model
            device: Device to train on
            temperature: Distillation temperature
            alpha: Weight for distillation loss
        """
        self.teacher_model = teacher_model.to(device).eval()
        self.student_model = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        self.loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
        self.temperature_scaling = TemperatureScaling()
        
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of metrics (loss, accuracy)
        """
        self.student_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)
            
            student_logits = self.student_model(inputs)
            
            # Compute loss
            loss = self.loss_fn(student_logits, teacher_logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(student_logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        self.metrics["train_losses"].append(avg_loss)
        self.metrics["train_accs"].append(accuracy)
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set.
        
        Returns:
            Dictionary of metrics (loss, accuracy)
        """
        self.student_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                teacher_logits = self.teacher_model(inputs)
                student_logits = self.student_model(inputs)
                
                # Compute loss
                loss = self.loss_fn(student_logits, teacher_logits, targets)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(student_logits, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        self.metrics["val_losses"].append(avg_loss)
        self.metrics["val_accs"].append(accuracy)
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
    ) -> Dict[str, list]:
        """Train student model.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Early stopping
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                      f"Val Acc: {val_metrics['val_accuracy']:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            **self.metrics,
            "training_time_s": training_time,
        }

    def compare_models(self) -> Dict[str, float]:
        """Compare student and teacher model sizes and latency.
        
        Returns:
            Dictionary of comparison metrics
        """
        teacher_size = self._calculate_model_size(self.teacher_model)
        student_size = self._calculate_model_size(self.student_model)
        
        return {
            "teacher_size_mb": teacher_size / (1024 * 1024),
            "student_size_mb": student_size / (1024 * 1024),
            "compression_ratio": teacher_size / student_size,
        }

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    def get_metrics(self) -> Dict:
        """Get training metrics.
        
        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()
