"""
Meta-Learning Framework for Few-Shot Learning

This module provides base classes and utilities for meta-learning approaches
including MAML, Prototypical Networks, and Matching Networks.

Key Concepts:
- Meta-learning: Learning to learn (learn good initialization)
- Few-shot learning: Adapt to new task with minimal data (1-5 examples)
- Support set: Training examples for task
- Query set: Test examples for task
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np


@dataclass
class TaskBatch:
    """Represents a batch of few-shot learning tasks."""
    support_x: torch.Tensor  # Shape: (num_tasks, num_support, *feature_shape)
    support_y: torch.Tensor  # Shape: (num_tasks, num_support)
    query_x: torch.Tensor    # Shape: (num_tasks, num_query, *feature_shape)
    query_y: torch.Tensor    # Shape: (num_tasks, num_query)
    
    def to(self, device: torch.device) -> "TaskBatch":
        """Move tensors to device."""
        return TaskBatch(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device)
        )


class AbstractMetaLearner(ABC, nn.Module):
    """
    Base class for meta-learning algorithms.
    
    All meta-learners implement:
    1. Meta-training: Learn good initialization across tasks
    2. Meta-testing: Adapt to new tasks with few examples
    3. Few-shot learning: Support set → adapted model → query prediction
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize meta-learner.
        
        Args:
            model: Underlying neural network to meta-train
            device: Device to use (cuda/cpu)
        """
        super().__init__()
        self.model = model
        self.device = device or torch.device('cpu')
        self.to(self.device)
    
    @abstractmethod
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None:
        """
        Adapt model to new task using support set.
        
        This is the inner loop: use support set to specialize model
        for new task with minimal examples.
        
        Args:
            support_x: Support set features (num_support, *feature_shape)
            support_y: Support set labels (num_support,)
        """
        pass
    
    @abstractmethod
    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on query set after adaptation.
        
        Args:
            query_x: Query set features (num_query, *feature_shape)
            
        Returns:
            Predictions (num_query, num_classes)
        """
        pass
    
    @abstractmethod
    def meta_update(self, tasks: TaskBatch, meta_optimizer: Optimizer) -> Dict[str, float]:
        """
        Outer loop: Update meta-parameters based on task batch.
        
        This is the meta-update: use adapted model's performance
        on multiple tasks to update meta-parameters.
        
        Args:
            tasks: Batch of few-shot tasks
            meta_optimizer: Optimizer for meta-parameters
            
        Returns:
            Dictionary of metrics (loss, accuracy, etc.)
        """
        pass
    
    def forward(self, tasks: TaskBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass through meta-learner.
        
        Args:
            tasks: Batch of few-shot tasks
            
        Returns:
            - Predictions on query sets
            - Metrics dictionary
        """
        batch_size = tasks.support_x.shape[0]
        all_predictions = []
        all_labels = []
        
        for i in range(batch_size):
            # Adapt to task i
            self.adapt(tasks.support_x[i], tasks.support_y[i])
            
            # Predict on query set
            pred = self.predict(tasks.query_x[i])
            all_predictions.append(pred)
            all_labels.append(tasks.query_y[i])
        
        predictions = torch.stack(all_predictions, dim=0)
        labels = torch.stack(all_labels, dim=0)
        
        return predictions, labels


class TaskSampler:
    """
    Samples few-shot learning tasks from dataset.
    
    A few-shot task consists of:
    - Support set: few examples per class (e.g., 5 examples × 5 ways = 25 total)
    - Query set: test examples per class (e.g., 10 examples × 5 ways = 50 total)
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_ways: int = 5,
        num_shots: int = 5,
        num_queries: int = 10,
        num_tasks: int = 32,
        random_seed: int = None
    ):
        """
        Initialize task sampler.
        
        Args:
            dataset: Dataset to sample from
            num_ways: Number of classes per task
            num_shots: Number of support examples per class
            num_queries: Number of query examples per class
            num_tasks: Number of tasks per batch
            random_seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_tasks = num_tasks
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # Group indices by class
        self._organize_by_class()
    
    def _organize_by_class(self):
        """Organize dataset indices by class."""
        self.class_indices = {}
        
        for idx, (_, label) in enumerate(self.dataset):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def sample_task(self) -> TaskBatch:
        """
        Sample a single few-shot task.
        
        Returns:
            TaskBatch with support and query sets
        """
        # Sample num_ways classes
        selected_classes = np.random.choice(
            list(self.class_indices.keys()),
            size=self.num_ways,
            replace=False
        )
        
        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []
        
        for class_id, class_label in enumerate(selected_classes):
            # Sample support + query examples
            available_indices = self.class_indices[class_label]
            sampled = np.random.choice(
                available_indices,
                size=self.num_shots + self.num_queries,
                replace=False
            )
            
            support_indices.extend(sampled[:self.num_shots])
            query_indices.extend(sampled[self.num_shots:])
            
            support_labels.extend([class_id] * self.num_shots)
            query_labels.extend([class_id] * self.num_queries)
        
        # Load features and labels
        support_features = []
        for idx in support_indices:
            feature, _ = self.dataset[idx]
            support_features.append(feature)
        support_features = torch.stack(support_features)
        
        query_features = []
        for idx in query_indices:
            feature, _ = self.dataset[idx]
            query_features.append(feature)
        query_features = torch.stack(query_features)
        
        return TaskBatch(
            support_x=support_features,
            support_y=torch.tensor(support_labels, dtype=torch.long),
            query_x=query_features,
            query_y=torch.tensor(query_labels, dtype=torch.long)
        )
    
    def sample_batch(self) -> TaskBatch:
        """
        Sample a batch of tasks.
        
        Returns:
            TaskBatch with all tasks stacked
        """
        tasks = [self.sample_task() for _ in range(self.num_tasks)]
        
        # Stack tasks into batch dimension
        return TaskBatch(
            support_x=torch.stack([t.support_x for t in tasks]),
            support_y=torch.stack([t.support_y for t in tasks]),
            query_x=torch.stack([t.query_x for t in tasks]),
            query_y=torch.stack([t.query_y for t in tasks])
        )


class FeatureExtractor(nn.Module):
    """
    Simple feature extraction backbone for meta-learning.
    
    Typical architecture:
    - Conv2d blocks with ReLU
    - MaxPooling for spatial reduction
    - Output: flattened features
    """
    
    def __init__(self, input_channels: int = 1, hidden_dim: int = 64):
        """
        Initialize feature extractor.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Feature dimension after conv
        self.output_dim = hidden_dim * 1 * 1  # After 4x downsampling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.
        
        Args:
            x: Input tensor
            
        Returns:
            Flattened feature vector
        """
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return x


def clone_model(model: nn.Module) -> nn.Module:
    """
    Create a deep copy of model with same architecture and weights.
    
    Used in MAML for task-specific model adaptation.
    
    Args:
        model: Model to clone
        
    Returns:
        Cloned model with same parameters
    """
    cloned = type(model)(*[p.clone().detach() for p in model.parameters()])
    cloned.load_state_dict(model.state_dict())
    return cloned


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted logits (batch_size, num_classes)
        labels: True labels (batch_size,)
        
    Returns:
        Accuracy as float in [0, 1]
    """
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == labels).float().sum().item()
    return correct / len(labels)
