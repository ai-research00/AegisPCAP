"""
Meta-Learning & Transfer Learning Module for AegisPCAP Phase 12

Provides few-shot learning, domain adaptation, multi-task learning,
and continual learning capabilities for advanced threat detection.

Author: AegisPCAP Development
Date: February 5, 2026
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import pickle

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Data Structures & Configuration
# ============================================================================

@dataclass
class FewShotBatch:
    """Batch structure for few-shot learning tasks."""
    support_x: torch.Tensor  # Shape: (n_way, k_shot, n_features)
    support_y: torch.Tensor  # Shape: (n_way, k_shot)
    query_x: torch.Tensor    # Shape: (n_way, q_query, n_features)
    query_y: torch.Tensor    # Shape: (n_way, q_query)
    domain_id: Optional[int] = None


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning models."""
    n_way: int = 5              # Number of attack classes per task
    k_shot: int = 5             # Number of support examples per class
    q_query: int = 15           # Number of query examples per class
    feature_dim: int = 50       # Feature dimension
    embedding_dim: int = 64     # Embedding dimension
    learning_rate: float = 0.001
    meta_lr: float = 0.0001
    inner_loops: int = 5        # MAML inner loop iterations
    outer_loops: int = 10       # MAML outer loop iterations


# ============================================================================
# Few-Shot Learning: Prototypical Networks
# ============================================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning.
    
    Paper: "Prototypical Networks for Few-shot Learning"
    https://arxiv.org/abs/1703.05175
    
    Learns a metric space where classification can be performed by
    computing distances to prototype representations of each class.
    """
    
    def __init__(self, feature_dim: int, embedding_dim: int):
        """
        Args:
            feature_dim: Input feature dimension
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Learnable embedding network
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space."""
        return self.encoder(x)
    
    def compute_prototypes(self, support_x: torch.Tensor, 
                          support_y: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Args:
            support_x: Shape (n_way, k_shot, feature_dim)
            support_y: Shape (n_way, k_shot)
            
        Returns:
            prototypes: Shape (n_way, embedding_dim)
        """
        n_way = support_x.size(0)
        embeddings = self.forward(support_x.reshape(-1, self.feature_dim))
        embeddings = embeddings.reshape(n_way, -1, self.embedding_dim)
        
        # Average embeddings per class
        prototypes = embeddings.mean(dim=1)
        return prototypes
    
    def predict_query(self, query_x: torch.Tensor, 
                     prototypes: torch.Tensor,
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Classify query samples using prototypes.
        
        Args:
            query_x: Shape (n_query, feature_dim)
            prototypes: Shape (n_way, embedding_dim)
            temperature: Softmax temperature for regularization
            
        Returns:
            logits: Shape (n_query, n_way)
        """
        query_embeddings = self.forward(query_x)
        
        # Compute distances (negative distance as logits)
        distances = torch.cdist(query_embeddings, prototypes)  # (n_query, n_way)
        logits = -distances / temperature
        return logits


class FewShotLearner:
    """
    Few-Shot Learning wrapper for threat detection.
    
    Enables classification of new attack types with minimal training data.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize few-shot learner."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = PrototypicalNetwork(
            config.feature_dim,
            config.embedding_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_batch(self, batch: FewShotBatch) -> float:
        """
        Train on a single few-shot batch.
        
        Args:
            batch: FewShotBatch with support and query sets
            
        Returns:
            loss: Training loss value
        """
        self.model.train()
        
        support_x = batch.support_x.to(self.device)
        support_y = batch.support_y.to(self.device)
        query_x = batch.query_x.to(self.device)
        query_y = batch.query_y.to(self.device)
        
        # Compute prototypes from support set
        prototypes = self.model.compute_prototypes(support_x, support_y)
        
        # Classify query samples
        logits = self.model.predict_query(
            query_x.reshape(-1, self.config.feature_dim),
            prototypes
        )
        
        # Compute loss
        loss = self.criterion(logits, query_y.reshape(-1).long())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, support_x: np.ndarray, support_y: np.ndarray,
               query_x: np.ndarray) -> np.ndarray:
        """
        Predict labels for query samples given support set.
        
        Args:
            support_x: Shape (n_way, k_shot, feature_dim)
            support_y: Shape (n_way, k_shot)
            query_x: Shape (n_query, feature_dim)
            
        Returns:
            predictions: Shape (n_query,) class indices
        """
        self.model.eval()
        
        support_x_t = torch.FloatTensor(support_x).to(self.device)
        query_x_t = torch.FloatTensor(query_x).to(self.device)
        
        with torch.no_grad():
            prototypes = self.model.compute_prototypes(support_x_t, 
                                                       torch.zeros(support_x.shape[0]))
            logits = self.model.predict_query(query_x_t, prototypes)
            predictions = logits.argmax(dim=1).cpu().numpy()
        
        return predictions


# ============================================================================
# Domain Adaptation
# ============================================================================

class DomainAdaptationLayer(nn.Module):
    """
    Domain adaptation layer for cross-network transfer.
    
    Uses adversarial training to learn domain-invariant features.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Feature extractor (domain-invariant)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Domain classifier (adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary: source or target domain
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary: threat or benign
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract domain-invariant features."""
        return self.feature_extractor(x)
    
    def classify_domain(self, features: torch.Tensor) -> torch.Tensor:
        """Classify domain (adversarial)."""
        return self.domain_classifier(features)
    
    def classify_task(self, features: torch.Tensor) -> torch.Tensor:
        """Classify task (threat detection)."""
        return self.task_classifier(features)


class DomainAdaptor:
    """Domain adaptation for cross-network threat detection."""
    
    def __init__(self, feature_dim: int, lambda_adversarial: float = 0.1):
        """
        Args:
            feature_dim: Feature dimension
            lambda_adversarial: Weight for adversarial loss
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DomainAdaptationLayer(feature_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.lambda_adversarial = lambda_adversarial
        
        self.task_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def train_step(self, source_x: torch.Tensor, source_y: torch.Tensor,
                  target_x: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for domain adaptation.
        
        Args:
            source_x: Source domain features
            source_y: Source domain labels
            target_x: Target domain features (unlabeled)
            
        Returns:
            losses: Dictionary of loss values
        """
        self.model.train()
        
        source_x = source_x.to(self.device)
        source_y = source_y.to(self.device).float()
        target_x = target_x.to(self.device)
        
        # Extract features
        source_features = self.model.extract_features(source_x)
        target_features = self.model.extract_features(target_x)
        
        # Task loss (source domain)
        task_logits = self.model.classify_task(source_features)
        task_loss = self.task_criterion(task_logits.squeeze(), source_y)
        
        # Domain loss (adversarial)
        source_domain_logits = self.model.classify_domain(source_features)
        target_domain_logits = self.model.classify_domain(target_features)
        
        source_domain_labels = torch.zeros(source_features.size(0), dtype=torch.long)
        target_domain_labels = torch.ones(target_features.size(0), dtype=torch.long)
        
        source_domain_labels = source_domain_labels.to(self.device)
        target_domain_labels = target_domain_labels.to(self.device)
        
        domain_loss = (
            self.domain_criterion(source_domain_logits, source_domain_labels) +
            self.domain_criterion(target_domain_logits, target_domain_labels)
        )
        
        # Total loss (task loss - lambda * domain loss for min-max)
        total_loss = task_loss - self.lambda_adversarial * domain_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "task_loss": task_loss.item(),
            "domain_loss": domain_loss.item(),
            "total_loss": total_loss.item()
        }


# ============================================================================
# Multi-Task Learning
# ============================================================================

class MultiTaskLearner(nn.Module):
    """
    Multi-Task Learning for simultaneous threat detection.
    
    Detects multiple threats (C2, exfiltration, ransomware, etc.)
    with shared representations and task-specific heads.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                num_tasks: int = 5):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Shared representation dimension
            num_tasks: Number of detection tasks
        """
        super().__init__()
        self.num_tasks = num_tasks
        
        # Shared encoder (learns common representations)
        self.shared_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Binary classification per task
            )
            for _ in range(num_tasks)
        ])
        
        # Task-specific attention
        self.task_attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_tasks),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Args:
            x: Input features of shape (batch_size, feature_dim)
            
        Returns:
            task_outputs: List of task logits
        """
        shared_repr = self.shared_encoder(x)
        task_outputs = [head(shared_repr) for head in self.task_heads]
        return task_outputs
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with task attention weights.
        
        Args:
            x: Input features
            
        Returns:
            task_outputs: Task predictions
            attention_weights: Attention weights per task
        """
        shared_repr = self.shared_encoder(x)
        task_outputs = [head(shared_repr) for head in self.task_heads]
        attention_weights = self.task_attention(shared_repr)
        return task_outputs, attention_weights


class MultiTaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self, feature_dim: int, num_tasks: int = 5,
                task_weights: Optional[List[float]] = None):
        """
        Args:
            feature_dim: Input feature dimension
            num_tasks: Number of tasks
            task_weights: Loss weights per task
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTaskLearner(feature_dim, num_tasks=num_tasks).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Task weights for loss balancing
        self.task_weights = task_weights or [1.0] * num_tasks
        
    def train_batch(self, x: torch.Tensor, 
                   task_labels: List[torch.Tensor]) -> float:
        """
        Train on a batch with multiple task labels.
        
        Args:
            x: Input features
            task_labels: List of labels per task
            
        Returns:
            total_loss: Combined loss across all tasks
        """
        self.model.train()
        
        x = x.to(self.device)
        task_labels = [y.to(self.device) for y in task_labels]
        
        # Forward pass
        task_outputs = self.model(x)
        
        # Compute weighted loss
        total_loss = 0.0
        for i, (output, labels) in enumerate(zip(task_outputs, task_labels)):
            task_loss = self.criterion(output.squeeze(), labels.float())
            total_loss += self.task_weights[i] * task_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# ============================================================================
# Continual Learning: Elastic Weight Consolidation
# ============================================================================

class ContinualLearner:
    """
    Continual Learning with Elastic Weight Consolidation (EWC).
    
    Prevents catastrophic forgetting when learning new attack types
    by consolidating important weights from previous tasks.
    
    Paper: "Overcoming catastrophic forgetting in neural networks"
    https://arxiv.org/abs/1612.00796
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4,
                replay_buffer_size: int = 1000):
        """
        Args:
            model: Neural network model
            lambda_ewc: EWC regularization strength
            replay_buffer_size: Size of experience replay buffer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lambda_ewc = lambda_ewc
        
        # Fisher Information Matrix (importance of weights)
        self.fisher_information = None
        self.optimal_weights = None
        
        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def compute_fisher_information(self, dataloader, num_samples: int = 1000):
        """
        Compute Fisher Information Matrix for current task.
        
        Estimates parameter importance for EWC.
        
        Args:
            dataloader: Data loader for current task
            num_samples: Number of samples for estimation
        """
        logger.info(f"Computing Fisher Information Matrix ({num_samples} samples)...")
        
        self.model.eval()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch_x, batch_y in dataloader:
            if sample_count >= num_samples:
                break
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y.long())
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            sample_count += batch_x.size(0)
        
        # Normalize by samples
        for name in fisher:
            fisher[name] /= sample_count
        
        self.fisher_information = fisher
        self.optimal_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        logger.info("Fisher Information Matrix computed.")
    
    def train_with_ewc(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Train with EWC regularization to prevent forgetting.
        
        Args:
            batch_x: Input batch
            batch_y: Labels
            
        Returns:
            loss: Training loss with EWC regularization
        """
        self.model.train()
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        # Standard training loss
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y.long())
        
        # EWC regularization (if previous task learned)
        if self.fisher_information is not None and self.optimal_weights is not None:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fisher_information:
                    fisher = self.fisher_information[name]
                    optimal_weight = self.optimal_weights[name]
                    ewc_loss += (fisher * (param - optimal_weight) ** 2).sum()
            
            loss += self.lambda_ewc * ewc_loss
        
        # Store in replay buffer
        self.replay_buffer.append((batch_x.cpu().detach(), batch_y.cpu().detach()))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# Meta-Learning Controller
# ============================================================================

class MetaLearningController:
    """
    Unified controller for all meta-learning techniques.
    
    Coordinates few-shot learning, domain adaptation, multi-task learning,
    and continual learning for comprehensive threat detection.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize meta-learning controller."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize all meta-learning components
        self.few_shot_learner = FewShotLearner(config)
        self.domain_adaptor = DomainAdaptor(config.feature_dim)
        self.multi_task_learner = MultiTaskTrainer(config.feature_dim)
        
        # Continual learner wraps the multi-task model
        self.continual_learner = ContinualLearner(
            self.multi_task_learner.model,
            lambda_ewc=0.4
        )
        
        self.history = defaultdict(list)
        
    def add_new_attack_type(self, support_data: np.ndarray,
                           support_labels: np.ndarray) -> Dict:
        """
        Add new attack type using few-shot learning.
        
        Args:
            support_data: Support examples (k_shot, feature_dim)
            support_labels: Labels (k_shot,)
            
        Returns:
            result: Dictionary with training results
        """
        logger.info(f"Adding new attack type with {len(support_data)} support examples...")
        
        # Create few-shot batch
        n_way = len(np.unique(support_labels))
        batch = FewShotBatch(
            support_x=torch.FloatTensor(support_data[:n_way]).unsqueeze(1),
            support_y=torch.LongTensor(support_labels[:n_way]),
            query_x=torch.FloatTensor(support_data).unsqueeze(1),
            query_y=torch.LongTensor(support_labels)
        )
        
        # Train
        loss = self.few_shot_learner.train_batch(batch)
        
        result = {
            "attack_type": "new_attack",
            "support_count": len(support_data),
            "initial_loss": loss,
            "status": "trained"
        }
        
        self.history["new_attacks"].append(result)
        return result
    
    def adapt_to_domain(self, source_data: np.ndarray,
                       source_labels: np.ndarray,
                       target_data: np.ndarray) -> Dict:
        """
        Adapt model to new domain (network environment).
        
        Args:
            source_data: Source domain features
            source_labels: Source domain labels
            target_data: Target domain features (unlabeled)
            
        Returns:
            result: Adaptation results
        """
        logger.info(f"Adapting to new domain ({len(target_data)} target samples)...")
        
        source_x = torch.FloatTensor(source_data)
        source_y = torch.LongTensor(source_labels)
        target_x = torch.FloatTensor(target_data)
        
        losses = []
        for epoch in range(5):  # Few adaptation rounds
            loss_dict = self.domain_adaptor.train_step(source_x, source_y, target_x)
            losses.append(loss_dict)
        
        result = {
            "source_samples": len(source_data),
            "target_samples": len(target_data),
            "adaptation_rounds": 5,
            "final_loss": losses[-1],
            "status": "adapted"
        }
        
        self.history["domain_adaptations"].append(result)
        return result
    
    def learn_new_threat_type_mtl(self, data: np.ndarray,
                                 task_labels: List[np.ndarray]) -> Dict:
        """
        Learn new threat type using multi-task learning.
        
        Args:
            data: Input features
            task_labels: Labels for each detection task
            
        Returns:
            result: Learning results
        """
        logger.info(f"Learning with multi-task approach ({len(task_labels)} tasks)...")
        
        x = torch.FloatTensor(data)
        task_labels_t = [torch.LongTensor(y) for y in task_labels]
        
        loss = self.multi_task_learner.train_batch(x, task_labels_t)
        
        result = {
            "tasks": len(task_labels),
            "samples": len(data),
            "mtl_loss": loss,
            "status": "learned"
        }
        
        self.history["mtl_training"].append(result)
        return result
    
    def learn_without_forgetting(self, data: np.ndarray,
                                labels: np.ndarray) -> Dict:
        """
        Learn new data without forgetting previous attacks.
        
        Uses Elastic Weight Consolidation.
        
        Args:
            data: New training data
            labels: New labels
            
        Returns:
            result: Continual learning results
        """
        logger.info(f"Continual learning: {len(data)} new samples...")
        
        x = torch.FloatTensor(data)
        y = torch.LongTensor(labels)
        
        losses = []
        for epoch in range(5):
            loss = self.continual_learner.train_with_ewc(x, y)
            losses.append(loss)
        
        result = {
            "new_samples": len(data),
            "training_epochs": 5,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "status": "continually_learned"
        }
        
        self.history["continual_learning"].append(result)
        return result
    
    def get_performance_summary(self) -> Dict:
        """Get summary of meta-learning performance."""
        return {
            "new_attacks_added": len(self.history["new_attacks"]),
            "domain_adaptations": len(self.history["domain_adaptations"]),
            "mtl_training_rounds": len(self.history["mtl_training"]),
            "continual_learning_rounds": len(self.history["continual_learning"]),
            "total_operations": sum(len(v) for v in self.history.values())
        }


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'MetaLearningConfig',
    'FewShotBatch',
    'PrototypicalNetwork',
    'FewShotLearner',
    'DomainAdaptationLayer',
    'DomainAdaptor',
    'MultiTaskLearner',
    'MultiTaskTrainer',
    'ContinualLearner',
    'MetaLearningController'
]

if __name__ == "__main__":
    print("Meta-Learning Module for AegisPCAP")
    print("Use: from ml.meta_learning import MetaLearningController")
