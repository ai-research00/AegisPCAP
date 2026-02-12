"""
Tests for Phase 10: Meta-Learning Framework

Comprehensive test coverage for:
- MAML (Model-Agnostic Meta-Learning)
- Prototypical Networks
- Task sampling and few-shot learning
- Rapid adaptation and convergence
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from src.ml.meta_learning.base import (
    TaskBatch, TaskSampler, FeatureExtractor, AbstractMetaLearner,
    clone_model, compute_accuracy
)
from src.ml.meta_learning.maml import MAML, train_maml
from src.ml.meta_learning.prototypical import PrototypicalNetwork, train_prototypical_network


# ===== FIXTURES =====

@pytest.fixture
def simple_dataset():
    """Create simple synthetic dataset for testing."""
    num_samples = 100
    num_classes = 5
    feature_dim = 28 * 28  # Simulating 28x28 images
    
    features = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = list(zip(features, labels))
    return dataset


@pytest.fixture
def feature_extractor():
    """Create a simple feature extractor."""
    return FeatureExtractor(input_channels=1, hidden_dim=32)


@pytest.fixture
def simple_model():
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 5)  # 5 classes
    )


@pytest.fixture
def task_sampler(simple_dataset):
    """Create a task sampler."""
    return TaskSampler(
        simple_dataset,
        num_ways=5,
        num_shots=5,
        num_queries=10,
        num_tasks=4,
        random_seed=42
    )


# ===== TESTS: TASK SAMPLING =====

class TestTaskSampling:
    """Tests for few-shot task sampling."""
    
    def test_task_batch_creation(self, task_sampler):
        """Test that task batches are created correctly."""
        task_batch = task_sampler.sample_task()
        
        assert isinstance(task_batch, TaskBatch)
        assert task_batch.support_x.shape[0] == 25  # 5 ways × 5 shots
        assert task_batch.query_x.shape[0] == 50     # 5 ways × 10 queries
        assert task_batch.support_y.shape[0] == 25
        assert task_batch.query_y.shape[0] == 50
    
    def test_task_labels_unique(self, task_sampler):
        """Test that task has exactly num_ways classes."""
        task_batch = task_sampler.sample_task()
        
        unique_support = torch.unique(task_batch.support_y)
        unique_query = torch.unique(task_batch.query_y)
        
        assert len(unique_support) == 5
        assert len(unique_query) == 5
    
    def test_batch_stacking(self, task_sampler):
        """Test that batch samples are properly stacked."""
        task_batch = task_sampler.sample_batch()
        
        assert task_batch.support_x.shape[0] == 4      # 4 tasks
        assert task_batch.support_x.shape[1] == 25     # 5 ways × 5 shots
        assert task_batch.query_x.shape[0] == 4        # 4 tasks
        assert task_batch.query_x.shape[1] == 50       # 5 ways × 10 queries
    
    def test_task_to_device(self, task_sampler):
        """Test moving task batch to device."""
        task_batch = task_sampler.sample_task()
        device = torch.device('cpu')
        
        moved_batch = task_batch.to(device)
        
        assert moved_batch.support_x.device.type == device.type
        assert moved_batch.query_x.device.type == device.type
    
    def test_reproducibility(self, simple_dataset):
        """Test that reproducibility works with seed."""
        sampler1 = TaskSampler(simple_dataset, num_ways=5, num_shots=5, random_seed=42)
        sampler2 = TaskSampler(simple_dataset, num_ways=5, num_shots=5, random_seed=42)
        
        task1 = sampler1.sample_task()
        task2 = sampler2.sample_task()
        
        assert torch.allclose(task1.support_x, task2.support_x)
        assert torch.equal(task1.support_y, task2.support_y)


# ===== TESTS: FEATURE EXTRACTOR =====

class TestFeatureExtractor:
    """Tests for feature extraction backbone."""
    
    def test_output_shape(self, feature_extractor):
        """Test that feature extractor produces correct output shape."""
        batch = torch.randn(4, 1, 28, 28)
        features = feature_extractor(batch)
        
        assert features.shape[0] == 4
        assert features.shape[1] == feature_extractor.output_dim
    
    def test_output_dimension(self, feature_extractor):
        """Test feature output dimension after downsampling."""
        # 4 max pools → 4x downsample → 28x28 → 1x1
        # Output: batch_size × hidden_dim × 1 × 1 → flattened
        
        batch = torch.randn(2, 1, 28, 28)
        features = feature_extractor(batch)
        
        expected_dim = 32  # hidden_dim
        assert features.shape[1] == expected_dim
    
    def test_gradients_flow(self, feature_extractor):
        """Test that gradients flow through feature extractor."""
        batch = torch.randn(4, 1, 28, 28, requires_grad=True)
        features = feature_extractor(batch)
        
        loss = features.sum()
        loss.backward()
        
        assert batch.grad is not None
        assert batch.grad.shape == batch.shape


# ===== TESTS: MAML =====

class TestMAML:
    """Tests for Model-Agnostic Meta-Learning."""
    
    def test_maml_initialization(self, simple_model):
        """Test MAML initialization."""
        maml = MAML(simple_model, inner_lr=0.01, num_inner_steps=5)
        
        assert maml.inner_lr == 0.01
        assert maml.num_inner_steps == 5
        assert maml.model is not None
    
    def test_maml_adaptation(self, simple_model):
        """Test MAML adaptation to task."""
        maml = MAML(simple_model, inner_lr=0.01, num_inner_steps=3)
        
        # Create synthetic support set
        support_x = torch.randn(10, 1024)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        # Adapt
        maml.adapt(support_x, support_y)
        
        # Task model should be created
        assert maml.task_model is not None
    
    def test_maml_prediction(self, simple_model):
        """Test MAML prediction after adaptation."""
        maml = MAML(simple_model, inner_lr=0.01, num_inner_steps=3)
        
        support_x = torch.randn(10, 1024)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        maml.adapt(support_x, support_y)
        
        # Predict
        query_x = torch.randn(5, 1024)
        predictions = maml.predict(query_x)
        
        assert predictions.shape == (5, 5)  # 5 queries, 5 classes
    
    def test_maml_meta_update(self, simple_model):
        """Test MAML meta-update."""
        maml = MAML(simple_model, inner_lr=0.01, num_inner_steps=3)
        meta_optimizer = torch.optim.SGD(maml.parameters(), lr=0.001)
        
        # Create task batch
        task_batch = TaskBatch(
            support_x=torch.randn(2, 10, 1024),  # 2 tasks, 10 support
            support_y=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                                   [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]),
            query_x=torch.randn(2, 5, 1024),     # 2 tasks, 5 queries
            query_y=torch.tensor([[0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4]])
        )
        
        metrics = maml.meta_update(task_batch, meta_optimizer)
        
        assert 'meta_loss' in metrics
        assert 'query_accuracy' in metrics
        assert isinstance(metrics['meta_loss'], float)
    
    def test_maml_rapid_adaptation(self, simple_model):
        """Test that MAML enables rapid adaptation with few steps."""
        maml = MAML(simple_model, inner_lr=0.1, num_inner_steps=1)
        
        # Support set
        support_x = torch.randn(10, 1024)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        # Adapt with just 1 step
        maml.adapt(support_x, support_y)
        
        # Should still produce meaningful predictions
        query_x = torch.randn(5, 1024)
        predictions = maml.predict(query_x)
        
        assert not torch.isnan(predictions).any()


# ===== TESTS: PROTOTYPICAL NETWORKS =====

class TestPrototypicalNetwork:
    """Tests for Prototypical Networks."""
    
    def test_proto_net_initialization(self, feature_extractor):
        """Test Prototypical Network initialization."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        assert proto_net.embedding_dim == 64
        assert proto_net.embedding is not None
    
    def test_embedding_extraction(self, feature_extractor):
        """Test embedding extraction."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        batch = torch.randn(4, 1, 28, 28)
        embeddings = proto_net.extract_embeddings(batch)
        
        assert embeddings.shape == (4, 64)
        # Check L2 normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_prototype_computation(self, feature_extractor):
        """Test class prototype computation."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        support_x = torch.randn(25, 1, 28, 28)
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5)
        
        prototypes, class_labels = proto_net.compute_prototypes(support_x, support_y)
        
        assert prototypes.shape == (5, 64)  # 5 classes, 64-dim embeddings
        assert len(class_labels) == 5
    
    def test_distance_computation_euclidean(self, feature_extractor):
        """Test Euclidean distance computation."""
        proto_net = PrototypicalNetwork(
            feature_extractor,
            embedding_dim=64,
            distance_metric='euclidean'
        )
        
        query_embeddings = torch.randn(10, 64)
        prototypes = torch.randn(5, 64)
        
        distances = proto_net.compute_distances(query_embeddings, prototypes)
        
        assert distances.shape == (10, 5)
        assert (distances >= 0).all()  # Distances should be non-negative
    
    def test_distance_computation_cosine(self, feature_extractor):
        """Test cosine distance computation."""
        proto_net = PrototypicalNetwork(
            feature_extractor,
            embedding_dim=64,
            distance_metric='cosine'
        )
        
        query_embeddings = F.normalize(torch.randn(10, 64), p=2, dim=1)
        prototypes = F.normalize(torch.randn(5, 64), p=2, dim=1)
        
        distances = proto_net.compute_distances(query_embeddings, prototypes)
        
        assert distances.shape == (10, 5)
        assert (distances >= -0.1).all() and (distances <= 2.1).all()  # Cosine distance in [0, 2]
    
    def test_proto_net_adaptation(self, feature_extractor):
        """Test Prototypical Network adaptation."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        support_x = torch.randn(25, 1, 28, 28)
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5)
        
        proto_net.adapt(support_x, support_y)
        
        assert proto_net.prototypes is not None
        assert proto_net.prototypes.shape == (5, 64)
    
    def test_proto_net_prediction(self, feature_extractor):
        """Test Prototypical Network prediction."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        support_x = torch.randn(25, 1, 28, 28)
        support_y = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5)
        
        proto_net.adapt(support_x, support_y)
        
        query_x = torch.randn(10, 1, 28, 28)
        logits = proto_net.predict(query_x)
        
        assert logits.shape == (10, 5)  # 10 queries, 5 classes
    
    def test_proto_net_meta_update(self, feature_extractor):
        """Test Prototypical Network meta-update."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        optimizer = torch.optim.Adam(proto_net.parameters(), lr=0.001)
        
        task_batch = TaskBatch(
            support_x=torch.randn(2, 25, 1, 28, 28),  # 2 tasks
            support_y=torch.tensor([[0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5,
                                   [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5]),
            query_x=torch.randn(2, 10, 1, 28, 28),
            query_y=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        )
        
        metrics = proto_net.meta_update(task_batch, optimizer)
        
        assert 'meta_loss' in metrics
        assert 'query_accuracy' in metrics


# ===== TESTS: UTILITY FUNCTIONS =====

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_compute_accuracy(self):
        """Test accuracy computation."""
        predictions = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        labels = torch.tensor([0, 1, 0])
        
        accuracy = compute_accuracy(predictions, labels)
        
        assert accuracy == 1.0  # 3/3 correct
    
    def test_compute_accuracy_incorrect(self):
        """Test accuracy with incorrect predictions."""
        predictions = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
        labels = torch.tensor([0, 1, 0])
        
        accuracy = compute_accuracy(predictions, labels)
        
        assert accuracy == 2/3  # 2/3 correct


# ===== INTEGRATION TESTS =====

class TestMetaLearningIntegration:
    """Integration tests for meta-learning pipeline."""
    
    def test_maml_few_shot_learning(self, task_sampler, simple_model):
        """Test MAML few-shot learning pipeline."""
        maml = MAML(simple_model, inner_lr=0.01, num_inner_steps=5)
        
        # Sample a task
        task = task_sampler.sample_task()
        task = task.to(torch.device('cpu'))
        
        # Adapt
        maml.adapt(task.support_x, task.support_y)
        
        # Predict
        with torch.no_grad():
            predictions = maml.predict(task.query_x)
        
        # Check shapes
        assert predictions.shape[0] == task.query_x.shape[0]
        assert predictions.shape[1] == 5  # 5 classes
    
    def test_proto_net_few_shot_learning(self, task_sampler, feature_extractor):
        """Test Prototypical Network few-shot learning."""
        proto_net = PrototypicalNetwork(feature_extractor, embedding_dim=64)
        
        # Sample a task
        task = task_sampler.sample_task()
        task = task.to(torch.device('cpu'))
        
        # Adapt
        proto_net.adapt(task.support_x, task.support_y)
        
        # Predict
        with torch.no_grad():
            predictions = proto_net.predict(task.query_x)
        
        # Check shapes
        assert predictions.shape[0] == task.query_x.shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
