"""
Prototypical Networks for Few-Shot Learning

Prototypical Networks learn an embedding space where classification is performed
by computing distances to class prototypes (mean embeddings of class support examples).

Key insight: Few-shot learning as metric learning in embedding space
- Learn feature extractor that produces good embeddings
- Compute class prototype = mean of support embeddings
- Classify query = nearest prototype in embedding space

Reference: "Prototypical Networks for Few-shot Learning"
           Snell et al., 2017 (https://arxiv.org/abs/1703.05175)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from typing import Dict, Tuple, Optional
import numpy as np

from .base import AbstractMetaLearner, TaskBatch, FeatureExtractor, compute_accuracy


class PrototypicalNetwork(AbstractMetaLearner):
    """
    Prototypical Networks meta-learner.
    
    Classification via metric learning:
    1. Extract embeddings from support and query sets
    2. Compute class prototypes (mean support embeddings)
    3. Classify queries by distance to prototypes
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        embedding_dim: int = 64,
        distance_metric: str = 'euclidean',
        device: torch.device = None
    ):
        """
        Initialize Prototypical Network.
        
        Args:
            feature_extractor: Module to extract features
            embedding_dim: Dimension of embedding space
            distance_metric: Distance metric ('euclidean' or 'cosine')
            device: Device (cuda/cpu)
        """
        super().__init__(feature_extractor, device)
        
        self.feature_extractor = feature_extractor
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Embedding projection layer
        input_dim = feature_extractor.output_dim if hasattr(feature_extractor, 'output_dim') else 64
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Storage for prototypes during inference
        self.prototypes = None
        self.class_labels = None
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding vectors for input.
        
        Args:
            x: Input features
            
        Returns:
            Embedding vectors (batch_size, embedding_dim)
        """
        features = self.feature_extractor(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)  # L2 normalization
    
    def compute_prototypes(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes from support set.
        
        Prototype = mean embedding of support examples for each class.
        
        Args:
            support_x: Support features (num_support, *feature_shape)
            support_y: Support labels (num_support,)
            
        Returns:
            - Prototypes (num_classes, embedding_dim)
            - Class indices
        """
        embeddings = self.extract_embeddings(support_x)
        
        unique_classes = torch.unique(support_y)
        num_classes = len(unique_classes)
        
        prototypes = []
        for class_id in unique_classes:
            # Get embeddings for this class
            class_mask = support_y == class_id
            class_embeddings = embeddings[class_mask]
            
            # Compute mean embedding (prototype)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes), unique_classes
    
    def compute_distances(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from queries to prototypes.
        
        Args:
            query_embeddings: Query embeddings (num_queries, embedding_dim)
            prototypes: Class prototypes (num_classes, embedding_dim)
            
        Returns:
            Distances (num_queries, num_classes)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||q - p||^2
            # Efficient computation: ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q·p
            query_norm = (query_embeddings ** 2).sum(dim=1, keepdim=True)
            prototype_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()
            dot_product = torch.mm(query_embeddings, prototypes.t())
            
            distances = query_norm + prototype_norm - 2 * dot_product
            distances = torch.sqrt(distances.clamp(min=1e-8))
        
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # (Already L2 normalized embeddings)
            distances = 1 - torch.mm(query_embeddings, prototypes.t())
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None:
        """
        Adapt to task by computing class prototypes.
        
        Args:
            support_x: Support features
            support_y: Support labels
        """
        self.eval()
        with torch.no_grad():
            self.prototypes, self.class_labels = self.compute_prototypes(support_x, support_y)
    
    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        """
        Predict on query set using prototypes.
        
        Classification = nearest prototype (smallest distance).
        
        Args:
            query_x: Query features (num_queries, *feature_shape)
            
        Returns:
            Logits (num_queries, num_classes)
        """
        if self.prototypes is None:
            raise RuntimeError("Must call adapt() before predict()")
        
        self.eval()
        with torch.no_grad():
            query_embeddings = self.extract_embeddings(query_x)
            distances = self.compute_distances(query_embeddings, self.prototypes)
        
        # Convert distances to logits (lower distance = higher probability)
        logits = -distances  # Negative distance as logits
        
        return logits
    
    def meta_update(self, tasks: TaskBatch, meta_optimizer: Optimizer) -> Dict[str, float]:
        """
        Update embeddings using task batch.
        
        Optimization objective: minimize distance between query embeddings
        and correct class prototype, maximize distance from other prototypes.
        
        Args:
            tasks: Batch of few-shot tasks
            meta_optimizer: Optimizer
            
        Returns:
            Metrics (loss, accuracy)
        """
        meta_optimizer.zero_grad()
        
        num_tasks = tasks.support_x.shape[0]
        total_loss = 0.0
        total_accuracy = 0.0
        
        self.train()
        
        for task_id in range(num_tasks):
            # Compute prototypes for this task
            prototypes, class_labels = self.compute_prototypes(
                tasks.support_x[task_id],
                tasks.support_y[task_id]
            )
            
            # Extract query embeddings
            query_embeddings = self.extract_embeddings(tasks.query_x[task_id])
            
            # Compute distances and logits
            distances = self.compute_distances(query_embeddings, prototypes)
            logits = -distances
            
            # Classification loss
            loss = F.cross_entropy(logits, tasks.query_y[task_id])
            total_loss += loss
            
            # Compute accuracy
            accuracy = compute_accuracy(logits.detach(), tasks.query_y[task_id])
            total_accuracy += accuracy
        
        # Average loss
        meta_loss = total_loss / num_tasks
        
        # Backward pass
        meta_loss.backward()
        
        # Optimize
        meta_optimizer.step()
        
        avg_accuracy = total_accuracy / num_tasks
        
        return {
            'meta_loss': meta_loss.item(),
            'query_accuracy': avg_accuracy
        }
    
    def forward(self, tasks: TaskBatch) -> Dict:
        """
        Forward pass through prototypical network.
        
        Args:
            tasks: Batch of few-shot tasks
            
        Returns:
            Dictionary with predictions and accuracy
        """
        batch_size = tasks.support_x.shape[0]
        all_accuracies = []
        
        self.eval()
        
        for task_id in range(batch_size):
            # Adapt to task
            self.adapt(tasks.support_x[task_id], tasks.support_y[task_id])
            
            # Predict
            with torch.no_grad():
                logits = self.predict(tasks.query_x[task_id])
            
            # Compute accuracy
            accuracy = compute_accuracy(logits, tasks.query_y[task_id])
            all_accuracies.append(accuracy)
        
        return {
            'accuracy': sum(all_accuracies) / len(all_accuracies)
        }


def train_prototypical_network(
    feature_extractor: nn.Module,
    embedding_dim: int,
    train_tasks,
    val_tasks,
    num_epochs: int = 100,
    meta_lr: float = 0.001,
    device: torch.device = None
) -> Dict[str, list]:
    """
    Training loop for Prototypical Networks.
    
    Args:
        feature_extractor: Feature extraction backbone
        embedding_dim: Embedding space dimension
        train_tasks: Training task sampler
        val_tasks: Validation task sampler
        num_epochs: Number of meta-training epochs
        meta_lr: Meta-learning rate
        device: Device (cuda/cpu)
        
    Returns:
        Training history
    """
    device = device or torch.device('cpu')
    
    # Initialize Prototypical Network
    proto_net = PrototypicalNetwork(
        feature_extractor,
        embedding_dim=embedding_dim,
        device=device
    )
    
    # Optimizer
    optimizer = Adam(proto_net.parameters(), lr=meta_lr)
    
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
        
        metrics = proto_net.meta_update(train_batch, optimizer)
        history['train_loss'].append(metrics['meta_loss'])
        history['train_accuracy'].append(metrics['query_accuracy'])
        
        # Validation
        if (epoch + 1) % 10 == 0:
            val_batch = val_tasks.sample_batch()
            val_batch = val_batch.to(device)
            
            val_results = proto_net.forward(val_batch)
            val_accuracy = val_results['accuracy']
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {metrics['meta_loss']:.4f} | "
                  f"Train Acc: {metrics['query_accuracy']:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
    
    return history
