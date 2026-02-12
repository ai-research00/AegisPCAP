"""
Comprehensive Test Suite for Phase 12 Advanced ML Modules

Tests for meta-learning, XAI, uncertainty quantification, and
federated learning modules.

Author: AegisPCAP Development
Date: February 5, 2026
Version: 1.0
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import tempfile
import os

from ml.meta_learning import (
    MetaLearningController, MetaLearningConfig, FewShotBatch,
    PrototypicalNetwork, FewShotLearner
)
from ml.explainable_ai import (
    XAIController, SHAPExplainer, LIMEExplainer,
    CounterfactualExplainer, FeatureInteractionAnalyzer
)
from ml.uncertainty_quantification import (
    UncertaintyQuantificationController, BayesianPredictor,
    MCDropoutPredictor, OODDetector, PredictionWithUncertainty
)
from ml.federated_learning import (
    FederatedLearningController, FederatedServer,
    FederatedClient, ThreatIntelligence, RolloutStrategy
)
from api.phase12_integration import Phase12Integration, Phase12Configuration


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Generate synthetic threat detection data."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 200
    n_features = 50
    
    # Normal traffic
    normal_data = np.random.randn(n_samples // 2, n_features)
    normal_labels = np.zeros(n_samples // 2)
    
    # Malicious traffic
    malicious_data = np.random.randn(n_samples // 2, n_features) + 2.0
    malicious_labels = np.ones(n_samples // 2)
    
    X = np.vstack([normal_data, malicious_data])
    y = np.hstack([normal_labels, malicious_labels])
    
    return X, y


@pytest.fixture
def simple_model():
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(50, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )


# ============================================================================
# Meta-Learning Tests
# ============================================================================

class TestMetaLearning:
    """Test meta-learning functionality."""
    
    def test_prototypical_network_forward(self):
        """Test prototypical network forward pass."""
        proto_net = PrototypicalNetwork(feature_dim=50, embedding_dim=64)
        
        # Create sample batch
        support_x = torch.randn(5, 3, 50)  # 5 classes, 3 examples
        support_y = torch.arange(5)
        
        # Forward pass
        embeddings = proto_net(support_x.reshape(-1, 50))
        assert embeddings.shape == (15, 64)
    
    def test_few_shot_learner_training(self, synthetic_data):
        """Test few-shot learner training."""
        config = MetaLearningConfig(feature_dim=50)
        learner = FewShotLearner(config)
        
        # Create few-shot batch
        X, y = synthetic_data
        batch = FewShotBatch(
            support_x=torch.FloatTensor(X[:10]).reshape(2, 5, 50),
            support_y=torch.LongTensor([0, 1]),
            query_x=torch.FloatTensor(X[10:20]),
            query_y=torch.LongTensor([0] * 5 + [1] * 5)
        )
        
        # Train
        loss = learner.train_batch(batch)
        assert loss > 0
        assert isinstance(loss, float)
    
    def test_few_shot_prediction(self, synthetic_data):
        """Test few-shot learning prediction."""
        config = MetaLearningConfig(feature_dim=50)
        learner = FewShotLearner(config)
        
        X, y = synthetic_data
        support_x = X[:10].reshape(2, 5, 50)
        support_y = np.array([0, 1])
        query_x = X[10:20]
        
        predictions = learner.predict(support_x, support_y, query_x)
        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_meta_learning_controller(self, synthetic_data):
        """Test meta-learning controller."""
        config = MetaLearningConfig(feature_dim=50)
        controller = MetaLearningController(config)
        
        X, y = synthetic_data
        
        # Test few-shot learning
        result = controller.add_new_attack_type(X[:10], y[:10])
        assert result["status"] == "trained"
        assert result["support_count"] == 10
    
    def test_continual_learning(self, synthetic_data):
        """Test continual learning without catastrophic forgetting."""
        config = MetaLearningConfig(feature_dim=50)
        controller = MetaLearningController(config)
        
        X, y = synthetic_data
        
        result = controller.learn_without_forgetting(X[:50], y[:50])
        assert result["status"] == "continually_learned"
        assert result["final_loss"] < result["initial_loss"]


# ============================================================================
# XAI Tests
# ============================================================================

class TestXAI:
    """Test explainable AI functionality."""
    
    def test_shap_explainer(self, synthetic_data, simple_model):
        """Test SHAP explanations."""
        X, y = synthetic_data
        
        # Train simple model
        model = simple_model
        explainer = SHAPExplainer(model, X[:50])
        
        # Explain single instance
        explanation = explainer.explain(X[0])
        
        assert explanation.prediction >= 0
        assert 0 <= explanation.confidence <= 1
        assert len(explanation.feature_importances) > 0
        assert explanation.explanation_method == "shap"
    
    def test_lime_explainer(self, synthetic_data, simple_model):
        """Test LIME explanations."""
        X, y = synthetic_data
        
        explainer = LIMEExplainer(simple_model, X[:50])
        explanation = explainer.explain(X[0], num_features=10)
        
        assert explanation.prediction >= 0
        assert 0 <= explanation.confidence <= 1
        assert len(explanation.feature_importances) > 0
        assert explanation.explanation_method == "lime"
    
    def test_feature_interaction_analyzer(self, synthetic_data, simple_model):
        """Test feature interaction analysis."""
        X, y = synthetic_data
        
        analyzer = FeatureInteractionAnalyzer(simple_model, X[:100])
        interactions = analyzer.analyze_interactions(n_top_pairs=5)
        
        assert len(interactions) <= 5
        for feature_i, feature_j, h_stat in interactions:
            assert isinstance(h_stat, float)
            assert h_stat >= 0
    
    def test_xai_controller(self, synthetic_data, simple_model):
        """Test unified XAI controller."""
        X, y = synthetic_data
        
        controller = XAIController(simple_model, X[:100])
        
        explanations = controller.explain_prediction(X[0], methods=["shap", "lime"])
        assert len(explanations) >= 1
        
        importances = controller.get_feature_importance_summary(X[0])
        assert len(importances) > 0


# ============================================================================
# Uncertainty Quantification Tests
# ============================================================================

class TestUncertaintyQuantification:
    """Test uncertainty quantification functionality."""
    
    def test_bayesian_predictor(self, synthetic_data):
        """Test Bayesian prediction."""
        X, y = synthetic_data
        
        predictor = BayesianPredictor(input_dim=50)
        
        # Train
        for i in range(3):
            x_batch = torch.FloatTensor(X[i*10:(i+1)*10])
            y_batch = torch.FloatTensor(y[i*10:(i+1)*10])
            loss = predictor.train_batch(x_batch, y_batch)
            assert loss > 0
        
        # Predict with uncertainty
        pred = predictor.predict_with_uncertainty(X[0])
        assert isinstance(pred, PredictionWithUncertainty)
        assert 0 <= pred.confidence <= 1
        assert pred.total_uncertainty >= 0
    
    def test_mc_dropout_predictor(self, synthetic_data):
        """Test MC Dropout prediction."""
        X, y = synthetic_data
        
        predictor = MCDropoutPredictor(input_dim=50)
        
        # Train
        for i in range(3):
            x_batch = torch.FloatTensor(X[i*10:(i+1)*10])
            y_batch = torch.FloatTensor(y[i*10:(i+1)*10])
            loss = predictor.train_batch(x_batch, y_batch)
            assert loss > 0
        
        # Predict with uncertainty
        pred = predictor.predict_with_uncertainty(X[0])
        assert isinstance(pred, PredictionWithUncertainty)
        assert 0 <= pred.confidence <= 1
    
    def test_ood_detector(self, synthetic_data):
        """Test out-of-distribution detection."""
        X, y = synthetic_data
        
        detector = OODDetector(threshold=2.0)
        detector.fit_mahalanobis(X[:100])
        
        # Check in-distribution
        is_ood = detector.is_ood(X[0])
        assert isinstance(is_ood, bool)
        
        # Mahalanobis distance
        distance = detector.mahalanobis_distance(X[0])
        assert distance >= 0
    
    def test_uncertainty_controller(self, synthetic_data):
        """Test unified uncertainty controller."""
        X, y = synthetic_data
        
        controller = UncertaintyQuantificationController(input_dim=50)
        
        predictions = controller.predict_with_full_uncertainty(X[0])
        assert len(predictions) >= 1
        
        for method, pred in predictions.items():
            assert isinstance(pred, PredictionWithUncertainty)
            assert pred.is_reliable() in [True, False]


# ============================================================================
# Federated Learning Tests
# ============================================================================

class TestFederatedLearning:
    """Test federated learning functionality."""
    
    def test_federated_client(self, synthetic_data):
        """Test federated learning client."""
        X, y = synthetic_data
        
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        client = FederatedClient(
            "client_1",
            model,
            (X[:50], y[:50])
        )
        
        # Train local
        metrics = client.train_local(num_epochs=2)
        assert metrics["epochs"] == 2
        assert metrics["samples"] == 50
        assert metrics["avg_loss"] > 0
        
        # Get weights
        weights = client.get_weights()
        assert len(weights) > 0
    
    def test_federated_server(self, synthetic_data):
        """Test federated learning server."""
        X, y = synthetic_data
        
        # Create clients
        clients = {}
        for i in range(3):
            model = nn.Sequential(
                nn.Linear(50, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            clients[f"client_{i}"] = FederatedClient(
                f"client_{i}",
                model,
                (X[i*50:(i+1)*50], y[i*50:(i+1)*50])
            )
        
        # Create server
        global_model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        server = FederatedServer(global_model, clients)
        
        # Conduct round
        fed_round = server.conduct_round(client_fraction=1.0, num_local_epochs=2)
        assert fed_round.round_number == 1
        assert len(fed_round.participating_clients) == 3
        assert fed_round.metrics["aggregation_quality"] > 0
    
    def test_threat_intelligence_sharing(self):
        """Test threat intelligence sharing."""
        threat = ThreatIntelligence(
            attack_type="C2",
            signature="dns_query_entropy>7.5",
            confidence=0.95,
            affected_organizations=["org1", "org2"],
            mitigation_actions=["block_domain", "alert_soc"]
        )
        
        assert threat.attack_type == "C2"
        assert threat.confidence == 0.95
        assert len(threat.mitigation_actions) == 2
    
    def test_federated_learning_controller(self, synthetic_data):
        """Test federated learning controller."""
        X, y = synthetic_data
        
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        controller = FederatedLearningController(model, num_clients=3)
        
        # Prepare data
        data_dict = {
            "client_1": (X[:50], y[:50]),
            "client_2": (X[50:100], y[50:100]),
            "client_3": (X[100:150], y[100:150])
        }
        
        # Initialize
        clients = controller.initialize_clients(data_dict)
        assert len(clients) == 3
        
        # Run training
        result = controller.run_federated_training(num_rounds=2, local_epochs=2)
        assert result["status"] == "success"
        assert result["rounds_completed"] == 2
    
    def test_threat_sharing(self):
        """Test threat intelligence sharing workflow."""
        controller = FederatedLearningController(
            nn.Linear(50, 1),
            num_clients=3
        )
        
        threat_id = controller.share_threat_intelligence(
            attack_type="Ransomware",
            signature="file_encryption_pattern",
            confidence=0.88,
            affected_orgs=["org_a"],
            mitigations=["isolate_host"],
            organization="org_a"
        )
        
        assert isinstance(threat_id, str)
        assert len(threat_id) > 0


# ============================================================================
# Phase 12 Integration Tests
# ============================================================================

class TestPhase12Integration:
    """Test complete Phase 12 integration."""
    
    def test_phase12_initialization(self):
        """Test Phase 12 system initialization."""
        config = Phase12Configuration(
            enable_meta_learning=True,
            enable_xai=True,
            enable_uncertainty=True,
            enable_federated_learning=True
        )
        
        integration = Phase12Integration(config)
        
        status = integration.get_phase12_status()
        assert status["modules"]["meta_learning"]["enabled"]
        assert status["modules"]["xai"]["enabled"]
        assert status["modules"]["uncertainty_quantification"]["enabled"]
        assert status["modules"]["federated_learning"]["enabled"]
    
    def test_add_new_attack_type(self, synthetic_data):
        """Test adding new attack type."""
        config = Phase12Configuration()
        integration = Phase12Integration(config)
        
        X, y = synthetic_data
        result = integration.add_new_attack_type(X[:10], y[:10], "new_botnet")
        
        assert result["status"] == "trained"
        assert result["attack_name"] == "new_botnet"
    
    def test_adapt_to_domain(self, synthetic_data):
        """Test domain adaptation."""
        config = Phase12Configuration()
        integration = Phase12Integration(config)
        
        X, y = synthetic_data
        result = integration.adapt_to_network_domain(
            X[:50], y[:50],
            X[50:100],
            "datacenter_network"
        )
        
        assert result["status"] == "adapted"
        assert result["domain_name"] == "datacenter_network"
    
    def test_uncertainty_prediction(self, synthetic_data):
        """Test prediction with uncertainty."""
        config = Phase12Configuration()
        integration = Phase12Integration(config)
        
        X, y = synthetic_data
        result = integration.predict_with_uncertainty(X[0])
        
        assert result["status"] == "success"
        assert 0 <= result["confidence"] <= 1
        assert result["total_uncertainty"] >= 0
    
    def test_xai_initialization(self, synthetic_data, simple_model):
        """Test XAI initialization."""
        config = Phase12Configuration()
        integration = Phase12Integration(config)
        
        X, y = synthetic_data
        status = integration.initialize_xai(simple_model, X[:100])
        
        assert status["status"] == "success"
    
    def test_federated_workflow(self, synthetic_data):
        """Test complete federated learning workflow."""
        config = Phase12Configuration(num_federated_clients=3)
        integration = Phase12Integration(config)
        
        X, y = synthetic_data
        
        # Initialize
        data_dict = {
            "client_1": (X[:50], y[:50]),
            "client_2": (X[50:100], y[50:100]),
            "client_3": (X[100:150], y[100:150])
        }
        
        init_result = integration.initialize_federated_learning(data_dict)
        assert init_result["status"] == "success"
        
        # Train
        train_result = integration.run_federated_training(num_rounds=2)
        assert train_result["status"] == "success"
    
    def test_threat_sharing_workflow(self, synthetic_data):
        """Test threat intelligence sharing."""
        config = Phase12Configuration()
        integration = Phase12Integration(config)
        
        # Initialize federated learning first
        X, y = synthetic_data
        data_dict = {
            "client_1": (X[:50], y[:50])
        }
        integration.initialize_federated_learning(data_dict)
        
        # Share threat
        result = integration.share_threat_intelligence(
            attack_type="C2",
            signature="dns_entropy>7.0",
            confidence=0.92,
            affected_orgs=["org1"],
            mitigations=["block_dns"],
            organization="org1"
        )
        
        assert result["status"] == "success"
        assert "threat_id" in result


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
