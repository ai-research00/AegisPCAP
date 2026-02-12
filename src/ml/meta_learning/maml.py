"""
Model-Agnostic Meta-Learning (MAML)

MAML learns a model initialization that can be rapidly adapted to new tasks
with just a few gradient steps on a small support set.

Key insight: Meta-learning through inner and outer loop optimization
- Inner loop: Gradient steps on support set (task adaptation)
- Outer loop: Update meta-parameters based on query set performance

Reference: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
           Finn et al., 2017 (https://arxiv.org/abs/1703.03400)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from typing import Dict, Tuple, List
import copy

from .base import AbstractMetaLearner, TaskBatch, compute_accuracy


class MAML(AbstractMetaLearner):
    """
    Model-Agnostic Meta-Learning implementation.
    
    Two-level optimization:
    1. Inner loop: Adapt task-specific model on support set (1-5 steps)
    2. Outer loop: Update meta-parameters on query set performance
    
    The key is that we compute gradients through the inner loop adaptation,
    so the outer loop learns initializations that are easy to adapt.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        device: torch.device = None
    ):
        """
        Initialize MAML meta-learner.
        
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for inner loop (task adaptation)
            num_inner_steps: Number of gradient steps in inner loop
            device: Device (cuda/cpu)
        """
        super().__init__(model, device)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Create task-specific model copy (for inner loop)
        self.task_model = None
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None:
        """
        Inner loop: Adapt model to task using support set.
        
        Performs num_inner_steps gradient steps on support set.
        
        Args:
            support_x: Support set features (num_support, *feature_shape)
            support_y: Support set labels (num_support,)
        """
        # Clone meta-model for task-specific adaptation
        self.task_model = copy.deepcopy(self.model)
        self.task_model.train()
        
        # Create task-specific optimizer
        task_optimizer = SGD(self.task_model.parameters(), lr=self.inner_lr)
        
        # Inner loop: adapt on support set
        for step in range(self.num_inner_steps):
            task_optimizer.zero_grad()
            
            # Forward pass on support set
            logits = self.task_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient step
            task_optimizer.step()
    
    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        """
        Predict on query set using adapted task model.
        
        Args:
            query_x: Query set features (num_query, *feature_shape)
            
        Returns:
            Logits (num_query, num_classes)
        """
        if self.task_model is None:
            # No adaptation, use meta-model
            return self.model(query_x)
        
        self.task_model.eval()
        with torch.no_grad():
            logits = self.task_model(query_x)
        return logits
    
    def meta_update(self, tasks: TaskBatch, meta_optimizer: Optimizer) -> Dict[str, float]:
        """
        Outer loop: Update meta-parameters using task batch.
        
        For each task:
        1. Adapt task model on support set (inner loop)
        2. Compute loss on query set
        3. Accumulate gradients (through adaptation)
        4. Update meta-parameters
        
        This is the key insight of MAML: gradients through the inner loop
        update the meta-parameters towards good initializations.
        
        Args:
            tasks: Batch of few-shot tasks
            meta_optimizer: Optimizer for meta-parameters
            
        Returns:
            Metrics (loss, accuracy)
        """
        meta_optimizer.zero_grad()
        
        num_tasks = tasks.support_x.shape[0]
        total_loss = 0.0
        total_accuracy = 0.0
        
        for task_id in range(num_tasks):
            # Inner loop: adapt to this task
            self._inner_loop_step(
                tasks.support_x[task_id],
                tasks.support_y[task_id]
            )
            
            # Outer loop: compute loss on query set
            query_loss, query_accuracy = self._outer_loop_step(
                tasks.query_x[task_id],
                tasks.query_y[task_id]
            )
            
            total_loss += query_loss
            total_accuracy += query_accuracy
        
        # Average loss over tasks
        meta_loss = total_loss / num_tasks
        
        # Backward pass through all inner loops
        meta_loss.backward()
        
        # Update meta-parameters
        meta_optimizer.step()
        
        avg_accuracy = total_accuracy / num_tasks
        
        return {
            'meta_loss': meta_loss.item(),
            'query_accuracy': avg_accuracy
        }
    
    def _inner_loop_step(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None:
        """
        Inner loop: one-step gradient update on support set.
        
        This creates a new computation graph for the outer loop to differentiate through.
        
        Args:
            support_x: Support features
            support_y: Support labels
        """
        # Clone model for this task (requires_grad=True for outer loop)
        task_model = copy.deepcopy(self.model)
        
        # Forward pass
        logits = task_model(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        # Compute gradients (this graph will be used in outer loop)
        grads = torch.autograd.grad(
            loss,
            task_model.parameters(),
            create_graph=True,  # Keep computation graph for outer loop
            allow_unused=True
        )
        
        # Gradient step: update parameters
        with torch.no_grad():
            for param, grad in zip(task_model.parameters(), grads):
                if grad is not None:
                    param.data -= self.inner_lr * grad.data
        
        # Store adapted model for query set evaluation
        self.task_model = task_model
    
    def _outer_loop_step(self, query_x: torch.Tensor, query_y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Outer loop: evaluate on query set and compute loss.
        
        Args:
            query_x: Query features
            query_y: Query labels
            
        Returns:
            - Query loss (with computation graph)
            - Query accuracy
        """
        # Forward pass on adapted model
        logits = self.task_model(query_x)
        query_loss = F.cross_entropy(logits, query_y)
        
        # Compute accuracy (detached, for logging)
        query_accuracy = compute_accuracy(logits.detach(), query_y)
        
        return query_loss, query_accuracy
    
    def forward(self, tasks: TaskBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full forward pass through MAML meta-learning.
        
        Args:
            tasks: Batch of few-shot tasks
            
        Returns:
            - Predictions on query sets
            - Metrics
        """
        batch_size = tasks.support_x.shape[0]
        all_predictions = []
        all_labels = []
        all_accuracies = []
        
        for task_id in range(batch_size):
            # Adapt to this task
            self.adapt(tasks.support_x[task_id], tasks.support_y[task_id])
            
            # Predict on query set
            with torch.no_grad():
                predictions = self.predict(tasks.query_x[task_id])
            
            all_predictions.append(predictions)
            all_labels.append(tasks.query_y[task_id])
            
            # Compute accuracy
            accuracy = compute_accuracy(predictions, tasks.query_y[task_id])
            all_accuracies.append(accuracy)
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'accuracy': sum(all_accuracies) / len(all_accuracies)
        }


def train_maml(
    model: nn.Module,
    train_tasks,
    val_tasks,
    num_epochs: int = 100,
    meta_lr: float = 0.001,
    inner_lr: float = 0.01,
    num_inner_steps: int = 5,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Training loop for MAML meta-learning.
    
    Args:
        model: Base model architecture
        train_tasks: Training task sampler
        val_tasks: Validation task sampler
        num_epochs: Number of meta-training epochs
        meta_lr: Meta-learning rate (outer loop)
        inner_lr: Inner loop learning rate
        num_inner_steps: Number of inner loop steps
        device: Device (cuda/cpu)
        
    Returns:
        Dictionary with training history
    """
    device = device or torch.device('cpu')
    
    # Initialize MAML learner
    maml = MAML(model, inner_lr=inner_lr, num_inner_steps=num_inner_steps, device=device)
    
    # Meta-optimizer
    meta_optimizer = torch.optim.Adam(maml.parameters(), lr=meta_lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training batch
        train_batch = train_tasks.sample_batch()
        train_batch = train_batch.to(device)
        
        metrics = maml.meta_update(train_batch, meta_optimizer)
        history['train_loss'].append(metrics['meta_loss'])
        history['train_accuracy'].append(metrics['query_accuracy'])
        
        # Validation
        if (epoch + 1) % 10 == 0:
            val_batch = val_tasks.sample_batch()
            val_batch = val_batch.to(device)
            
            val_results = maml.forward(val_batch)
            val_accuracy = val_results['accuracy']
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {metrics['meta_loss']:.4f} | "
                  f"Train Acc: {metrics['query_accuracy']:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
    
    return history
