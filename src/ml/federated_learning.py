"""
Federated Learning Module for AegisPCAP Phase 12

Provides distributed training across organizations with privacy preservation,
collaborative threat intelligence sharing, and orchestration.

Author: AegisPCAP Development
Date: February 5, 2026
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import pickle
from abc import ABC, abstractmethod
from enum import Enum
import hashlib

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Enums & Data Structures
# ============================================================================

class RolloutStrategy(Enum):
    """Deployment strategies for federated models."""
    CANARY = "canary"        # Gradual rollout to subset
    BLUE_GREEN = "blue_green"  # Atomic switch between versions
    AB_TEST = "ab_test"      # A/B testing with traffic split


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    data_size: int
    model_version: str = "v1.0"
    is_active: bool = True


@dataclass
class FederatedRound:
    """Single round of federated learning."""
    round_number: int
    participating_clients: List[str]
    global_weights: Dict[str, torch.Tensor]
    client_weights: Dict[str, Dict[str, torch.Tensor]]  # client_id -> weights
    aggregation_method: str = "fedavg"  # Weighted average
    timestamp: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThreatIntelligence:
    """Threat intelligence to share across organization."""
    attack_type: str
    signature: str
    confidence: float
    affected_organizations: List[str]
    mitigation_actions: List[str]
    timestamp: Optional[str] = None


# ============================================================================
# Federated Averaging (FedAvg) Algorithm
# ============================================================================

class FederatedAverager:
    """
    Implements Federated Averaging (FedAvg) algorithm.
    
    Paper: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: Device to use ("cuda" or "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def aggregate(self, client_weights: Dict[str, Dict[str, torch.Tensor]],
                 client_data_sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client weights using weighted averaging.
        
        Args:
            client_weights: {client_id: {param_name: tensor}}
            client_data_sizes: {client_id: num_samples}
            
        Returns:
            aggregated_weights: Global aggregated weights
        """
        # Compute total samples
        total_samples = sum(client_data_sizes.values())
        
        aggregated = {}
        
        # Get list of parameter names from first client
        if not client_weights:
            return aggregated
        
        param_names = list(next(iter(client_weights.values())).keys())
        
        # Aggregate each parameter
        for param_name in param_names:
            weighted_param = torch.zeros_like(
                client_weights[list(client_weights.keys())[0]][param_name]
            )
            
            for client_id, weights in client_weights.items():
                client_weight = client_data_sizes[client_id] / total_samples
                weighted_param += client_weight * weights[param_name]
            
            aggregated[param_name] = weighted_param
        
        logger.info(f"Aggregated weights from {len(client_weights)} clients")
        return aggregated
    
    def compute_aggregation_quality(self, client_weights: Dict[str, Dict[str, torch.Tensor]],
                                    aggregated: Dict[str, torch.Tensor]) -> float:
        """
        Compute quality of aggregation (variance of client weights).
        
        Args:
            client_weights: Client weight dictionaries
            aggregated: Aggregated weights
            
        Returns:
            quality_score: Aggregation quality (0-1, higher is better)
        """
        total_variance = 0.0
        param_names = list(aggregated.keys())
        
        for param_name in param_names:
            # Compute variance of parameter across clients
            param_list = [
                weights[param_name]
                for weights in client_weights.values()
            ]
            
            if param_list:
                param_mean = torch.stack(param_list).mean(dim=0)
                variance = torch.stack([
                    (p - param_mean).pow(2).mean()
                    for p in param_list
                ]).mean()
                total_variance += variance.item()
        
        # Quality = 1 / (1 + variance)
        quality = 1.0 / (1.0 + total_variance)
        return quality


# ============================================================================
# Privacy-Preserving Aggregation
# ============================================================================

class DifferentialPrivacyMechanism:
    """Differential privacy for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_gaussian_noise(self, weights: Dict[str, torch.Tensor],
                         sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            weights: Model weights
            sensitivity: Sensitivity of query
            
        Returns:
            noisy_weights: Weights with DP noise
        """
        # Noise scale (Gaussian mechanism)
        sigma = (2 * sensitivity) / self.epsilon
        
        noisy_weights = {}
        for name, weight in weights.items():
            noise = torch.randn_like(weight) * sigma
            noisy_weights[name] = weight + noise
        
        logger.info(f"Applied DP noise (σ={sigma:.4f}, ε={self.epsilon})")
        return noisy_weights
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor],
                      norm_bound: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Clip gradients for DP (reduce sensitivity).
        
        Args:
            gradients: Model gradients
            norm_bound: Maximum norm
            
        Returns:
            clipped_gradients: Clipped gradients
        """
        total_norm = 0.0
        clipped = {}
        
        # Compute total norm
        for grad in gradients.values():
            total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip
        clip_factor = min(1.0, norm_bound / (total_norm + 1e-10))
        
        for name, grad in gradients.items():
            clipped[name] = grad * clip_factor
        
        return clipped


# ============================================================================
# Federated Learning Client
# ============================================================================

class FederatedClient:
    """Local federated learning client."""
    
    def __init__(self, client_id: str, model: nn.Module, train_data: Tuple,
                device: Optional[str] = None):
        """
        Args:
            client_id: Unique client identifier
            model: Local model
            train_data: (X, y) training data
            device: Computation device
        """
        self.client_id = client_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.local_epochs = 5
        self.batch_size = 32
        
    def train_local(self, num_epochs: int = None) -> Dict[str, float]:
        """
        Train model on local data.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            metrics: Training metrics
        """
        num_epochs = num_epochs or self.local_epochs
        
        self.model.train()
        X, y = self.train_data
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        losses = []
        for epoch in range(num_epochs):
            # Mini-batch training
            for i in range(0, len(X_t), self.batch_size):
                batch_x = X_t[i:i + self.batch_size]
                batch_y = y_t[i:i + self.batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        logger.info(f"Client {self.client_id}: Trained for {num_epochs} epochs, loss={avg_loss:.4f}")
        
        return {
            "client_id": self.client_id,
            "epochs": num_epochs,
            "avg_loss": avg_loss,
            "samples": len(X)
        }
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
    
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name].to(self.device))


# ============================================================================
# Federated Learning Server
# ============================================================================

class FederatedServer:
    """
    Central federated learning server.
    
    Orchestrates training rounds, aggregates client updates, distributes
    global model to clients.
    """
    
    def __init__(self, model: nn.Module, clients: Dict[str, FederatedClient],
                device: Optional[str] = None):
        """
        Args:
            model: Global model
            clients: {client_id: FederatedClient}
            device: Computation device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = model.to(self.device)
        self.clients = clients
        
        self.averager = FederatedAverager(device)
        self.dp_mechanism = DifferentialPrivacyMechanism(epsilon=2.0)
        
        self.round_history: List[FederatedRound] = []
        self.aggregation_quality_history: List[float] = []
        
    def conduct_round(self, client_fraction: float = 1.0,
                     num_local_epochs: int = 5,
                     apply_dp: bool = True) -> FederatedRound:
        """
        Conduct single federated learning round.
        
        Args:
            client_fraction: Fraction of clients to participate
            num_local_epochs: Local training epochs
            apply_dp: Apply differential privacy
            
        Returns:
            fed_round: FederatedRound with results
        """
        round_num = len(self.round_history) + 1
        
        # Select clients
        num_clients = max(1, int(len(self.clients) * client_fraction))
        selected_clients = np.random.choice(
            list(self.clients.keys()),
            size=num_clients,
            replace=False
        )
        
        logger.info(f"Round {round_num}: Selected {len(selected_clients)}/{len(self.clients)} clients")
        
        # Distribute global weights
        global_weights = {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
        
        for client_id in selected_clients:
            self.clients[client_id].set_weights(global_weights)
        
        # Collect client updates
        client_weights = {}
        client_data_sizes = {}
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Local training
            metrics = client.train_local(num_local_epochs)
            client_data_sizes[client_id] = metrics["samples"]
            
            # Get updated weights
            client_weights[client_id] = client.get_weights()
        
        # Apply privacy
        if apply_dp:
            # Clip and add noise to aggregated updates
            for client_id in client_weights:
                client_weights[client_id] = self.dp_mechanism.clip_gradients(
                    client_weights[client_id]
                )
        
        # Aggregate weights
        aggregated_weights = self.averager.aggregate(client_weights, client_data_sizes)
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_weights:
                    param.copy_(aggregated_weights[name].to(self.device))
        
        # Compute aggregation quality
        quality = self.averager.compute_aggregation_quality(client_weights, aggregated_weights)
        
        # Create round record
        fed_round = FederatedRound(
            round_number=round_num,
            participating_clients=list(selected_clients),
            global_weights=aggregated_weights,
            client_weights=client_weights,
            aggregation_method="fedavg_dp" if apply_dp else "fedavg",
            metrics={
                "num_clients": len(selected_clients),
                "aggregation_quality": quality,
                "avg_local_loss": np.mean([m["avg_loss"] for m in metrics])
            }
        )
        
        self.round_history.append(fed_round)
        self.aggregation_quality_history.append(quality)
        
        logger.info(f"Round {round_num} completed. Quality: {quality:.4f}")
        
        return fed_round
    
    def get_global_model(self) -> nn.Module:
        """Get current global model."""
        return self.global_model


# ============================================================================
# Model Versioning & Rollout
# ============================================================================

class ModelVersionRegistry:
    """Registry for federated model versions."""
    
    def __init__(self):
        """Initialize version registry."""
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.current_version = None
    
    def register_version(self, version_id: str, model: nn.Module,
                        metrics: Dict[str, float],
                        description: str = ""):
        """
        Register a new model version.
        
        Args:
            version_id: Semantic version (e.g., "v1.0.0")
            model: Model weights
            metrics: Performance metrics
            description: Version description
        """
        model_hash = hashlib.md5(
            pickle.dumps(model.state_dict())
        ).hexdigest()
        
        self.versions[version_id] = {
            "model": model.state_dict(),
            "metrics": metrics,
            "description": description,
            "hash": model_hash,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Registered model version: {version_id}")
    
    def rollout_strategy(self, from_version: str, to_version: str,
                        strategy: RolloutStrategy,
                        traffic_percentage: float = 100.0) -> Dict[str, Any]:
        """
        Plan rollout strategy for model version.
        
        Args:
            from_version: Current version
            to_version: Target version
            strategy: Rollout strategy
            traffic_percentage: Percentage of traffic to route to new version
            
        Returns:
            plan: Rollout plan
        """
        if from_version not in self.versions or to_version not in self.versions:
            raise ValueError("Version not found")
        
        plan = {
            "from_version": from_version,
            "to_version": to_version,
            "strategy": strategy.value,
            "traffic_percentage": traffic_percentage,
            "rollout_plan": []
        }
        
        if strategy == RolloutStrategy.CANARY:
            # Gradual: 10%, 25%, 50%, 100%
            plan["rollout_plan"] = [
                {"stage": 1, "traffic_percentage": 10, "duration_minutes": 5},
                {"stage": 2, "traffic_percentage": 25, "duration_minutes": 10},
                {"stage": 3, "traffic_percentage": 50, "duration_minutes": 15},
                {"stage": 4, "traffic_percentage": 100, "duration_minutes": 0}
            ]
        
        elif strategy == RolloutStrategy.BLUE_GREEN:
            # Atomic switch with rollback capability
            plan["rollout_plan"] = [
                {"stage": 1, "action": "deploy_new_version"},
                {"stage": 2, "action": "health_check"},
                {"stage": 3, "action": "switch_traffic"},
                {"stage": 4, "action": "monitor_stability"}
            ]
        
        elif strategy == RolloutStrategy.AB_TEST:
            # A/B test with statistical validation
            plan["rollout_plan"] = [
                {"stage": 1, "split": "50/50", "duration_hours": 24},
                {"stage": 2, "metric": "accuracy", "threshold": 0.02},
                {"stage": 3, "action": "determine_winner"}
            ]
        
        return plan


# ============================================================================
# Threat Intelligence Sharing
# ============================================================================

class ThreatIntelligenceHub:
    """Hub for sharing threat intelligence across organizations."""
    
    def __init__(self):
        """Initialize threat intelligence hub."""
        self.threat_repository: List[ThreatIntelligence] = []
        self.organization_submissions: defaultdict(list) = defaultdict(list)
    
    def submit_threat(self, threat: ThreatIntelligence, organization: str) -> str:
        """
        Submit threat intelligence.
        
        Args:
            threat: ThreatIntelligence object
            organization: Submitting organization
            
        Returns:
            threat_id: Assigned threat ID
        """
        threat_id = hashlib.md5(
            (threat.signature + threat.attack_type).encode()
        ).hexdigest()[:12]
        
        self.threat_repository.append(threat)
        self.organization_submissions[organization].append(threat_id)
        
        logger.info(f"Received threat from {organization}: {threat.attack_type}")
        
        return threat_id
    
    def get_collaborative_threats(self, organization: str,
                                 min_confidence: float = 0.7) -> List[ThreatIntelligence]:
        """
        Get threat intelligence for organization.
        
        Args:
            organization: Organization requesting threats
            min_confidence: Minimum confidence threshold
            
        Returns:
            threats: Relevant threats
        """
        relevant_threats = [
            t for t in self.threat_repository
            if (t.confidence >= min_confidence and
                organization not in t.affected_organizations)
        ]
        
        return relevant_threats
    
    def get_threat_coverage(self) -> Dict[str, int]:
        """Get coverage statistics."""
        return {
            "total_threats": len(self.threat_repository),
            "organizations": len(self.organization_submissions),
            "avg_confidence": np.mean([t.confidence for t in self.threat_repository]),
            "attack_types": len(set(t.attack_type for t in self.threat_repository))
        }


# ============================================================================
# Federated Learning Controller
# ============================================================================

class FederatedLearningController:
    """Unified controller for federated learning operations."""
    
    def __init__(self, model: nn.Module, num_clients: int):
        """Initialize federated learning controller."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.num_clients = num_clients
        
        self.server = None
        self.clients = {}
        self.threat_hub = ThreatIntelligenceHub()
        self.version_registry = ModelVersionRegistry()
    
    def initialize_clients(self, data_dict: Dict[str, Tuple]) -> Dict[str, FederatedClient]:
        """
        Initialize federated clients with local data.
        
        Args:
            data_dict: {client_id: (X, y)}
            
        Returns:
            clients: Dictionary of initialized clients
        """
        self.clients = {
            client_id: FederatedClient(
                client_id,
                self.model,  # Each client gets a copy
                data_dict[client_id],
                self.device
            )
            for client_id in data_dict.keys()
        }
        
        # Initialize server
        self.server = FederatedServer(self.model, self.clients, self.device)
        
        logger.info(f"Initialized {len(self.clients)} federated clients")
        return self.clients
    
    def run_federated_training(self, num_rounds: int,
                              client_fraction: float = 1.0,
                              local_epochs: int = 5) -> List[FederatedRound]:
        """
        Run complete federated training.
        
        Args:
            num_rounds: Number of federation rounds
            client_fraction: Fraction of clients per round
            local_epochs: Local training epochs
            
        Returns:
            history: List of FederatedRound results
        """
        logger.info(f"Starting federated training: {num_rounds} rounds, {len(self.clients)} clients")
        
        for round_num in range(num_rounds):
            fed_round = self.server.conduct_round(
                client_fraction=client_fraction,
                num_local_epochs=local_epochs,
                apply_dp=True
            )
            
            # Log round metrics
            metrics = fed_round.metrics
            logger.info(f"Round {round_num + 1}: Quality={metrics['aggregation_quality']:.4f}, "
                       f"Loss={metrics['avg_local_loss']:.4f}")
        
        return self.server.round_history
    
    def share_threat_intelligence(self, threat: ThreatIntelligence,
                                 organization: str) -> str:
        """Share threat with all organizations."""
        return self.threat_hub.submit_threat(threat, organization)
    
    def get_shared_threats(self, organization: str) -> List[ThreatIntelligence]:
        """Get shared threats for organization."""
        return self.threat_hub.get_collaborative_threats(organization)
    
    def register_model_version(self, version_id: str,
                              metrics: Dict[str, float]) -> None:
        """Register current global model as version."""
        self.version_registry.register_version(
            version_id,
            self.server.get_global_model(),
            metrics
        )
    
    def plan_rollout(self, from_version: str, to_version: str,
                    strategy: RolloutStrategy) -> Dict[str, Any]:
        """Plan model rollout."""
        return self.version_registry.rollout_strategy(from_version, to_version, strategy)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get federated training summary."""
        if not self.server:
            return {}
        
        return {
            "num_rounds": len(self.server.round_history),
            "num_clients": len(self.clients),
            "avg_aggregation_quality": float(np.mean(self.server.aggregation_quality_history))
            if self.server.aggregation_quality_history else 0.0,
            "threat_coverage": self.threat_hub.get_threat_coverage()
        }


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'RolloutStrategy',
    'ClientConfig',
    'FederatedRound',
    'ThreatIntelligence',
    'FederatedAverager',
    'DifferentialPrivacyMechanism',
    'FederatedClient',
    'FederatedServer',
    'ModelVersionRegistry',
    'ThreatIntelligenceHub',
    'FederatedLearningController'
]

if __name__ == "__main__":
    print("Federated Learning Module for AegisPCAP")
    print("Use: from ml.federated_learning import FederatedLearningController")
