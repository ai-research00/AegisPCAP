"""
Phase 12 Integration Layer for AegisPCAP

Integrates all advanced ML modules (meta-learning, XAI, uncertainty quantification,
federated learning) with existing PCAP pipeline and provides unified FastAPI endpoints.

Author: AegisPCAP Development
Date: February 5, 2026
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import json
from datetime import datetime

from ml.meta_learning import MetaLearningController, MetaLearningConfig
from ml.explainable_ai import XAIController
from ml.uncertainty_quantification import UncertaintyQuantificationController
from ml.federated_learning import FederatedLearningController, ThreatIntelligence, RolloutStrategy

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Phase 12 Integration Layer
# ============================================================================

@dataclass
class Phase12Configuration:
    """Configuration for Phase 12 advanced ML features."""
    enable_meta_learning: bool = True
    enable_xai: bool = True
    enable_uncertainty: bool = True
    enable_federated_learning: bool = True
    
    meta_learning_config: Optional[MetaLearningConfig] = None
    num_federated_clients: int = 5
    
    # Feature names for interpretability
    feature_names: Optional[List[str]] = None
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None


class Phase12Integration:
    """
    Unified integration of all Phase 12 advanced ML modules.
    
    Coordinates meta-learning, XAI, uncertainty quantification, and
    federated learning with existing AegisPCAP pipeline.
    """
    
    def __init__(self, config: Phase12Configuration):
        """Initialize Phase 12 integration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize modules
        self.meta_learning = None
        self.xai_controller = None
        self.uncertainty_controller = None
        self.federated_learning = None
        
        self._initialize_modules()
        
        # Tracking
        self.prediction_history: List[Dict[str, Any]] = []
        self.threat_sharing_history: List[Dict[str, Any]] = []
        self.federated_training_history: List[Dict[str, Any]] = []
    
    def _initialize_modules(self):
        """Initialize all Phase 12 modules."""
        if self.config.enable_meta_learning:
            ml_config = self.config.meta_learning_config or MetaLearningConfig()
            self.meta_learning = MetaLearningController(ml_config)
            logger.info("✅ Meta-Learning module initialized")
        
        if self.config.enable_xai:
            # XAI requires a trained model and data
            self.xai_controller = None  # Will be initialized with model
            logger.info("✅ XAI module ready for initialization")
        
        if self.config.enable_uncertainty:
            input_dim = self.config.meta_learning_config.feature_dim if self.config.meta_learning_config else 50
            self.uncertainty_controller = UncertaintyQuantificationController(input_dim)
            logger.info("✅ Uncertainty Quantification module initialized")
        
        if self.config.enable_federated_learning:
            # Create a simple model for federated learning
            simple_model = nn.Sequential(
                nn.Linear(50, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.federated_learning = FederatedLearningController(
                simple_model,
                self.config.num_federated_clients
            )
            logger.info("✅ Federated Learning module initialized")
    
    # ========================================================================
    # Meta-Learning Interface
    # ========================================================================
    
    def add_new_attack_type(self, support_data: np.ndarray,
                          support_labels: np.ndarray,
                          attack_name: str) -> Dict[str, Any]:
        """
        Add new attack type using few-shot learning.
        
        Args:
            support_data: Few examples of new attack
            support_labels: Labels for support examples
            attack_name: Name of the attack type
            
        Returns:
            result: Few-shot learning result
        """
        if self.meta_learning is None:
            return {"status": "error", "message": "Meta-learning not enabled"}
        
        logger.info(f"Adding new attack type: {attack_name} ({len(support_data)} examples)")
        
        result = self.meta_learning.add_new_attack_type(support_data, support_labels)
        result["attack_name"] = attack_name
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    
    def adapt_to_network_domain(self, source_data: np.ndarray,
                               source_labels: np.ndarray,
                               target_data: np.ndarray,
                               domain_name: str) -> Dict[str, Any]:
        """
        Adapt model to new network environment.
        
        Args:
            source_data: Source domain data (labeled)
            source_labels: Source labels
            target_data: Target domain data (unlabeled)
            domain_name: Name of target domain
            
        Returns:
            result: Adaptation result
        """
        if self.meta_learning is None:
            return {"status": "error", "message": "Meta-learning not enabled"}
        
        logger.info(f"Adapting to domain: {domain_name}")
        
        result = self.meta_learning.adapt_to_domain(
            source_data,
            source_labels,
            target_data
        )
        result["domain_name"] = domain_name
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    
    def enable_continual_learning(self, new_data: np.ndarray,
                                 new_labels: np.ndarray) -> Dict[str, Any]:
        """
        Continue learning on new data without forgetting.
        
        Args:
            new_data: New training data
            new_labels: New labels
            
        Returns:
            result: Continual learning result
        """
        if self.meta_learning is None:
            return {"status": "error", "message": "Meta-learning not enabled"}
        
        logger.info(f"Continual learning: {len(new_data)} new samples")
        
        result = self.meta_learning.learn_without_forgetting(new_data, new_labels)
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    
    # ========================================================================
    # XAI Interface
    # ========================================================================
    
    def initialize_xai(self, model: Any, training_data: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Initialize XAI with model and training data.
        
        Args:
            model: Trained prediction model
            training_data: Training dataset for SHAP/LIME
            feature_names: Feature names for interpretability
            
        Returns:
            status: Initialization status
        """
        if not self.config.enable_xai:
            return {"status": "error", "message": "XAI not enabled"}
        
        try:
            self.xai_controller = XAIController(
                model,
                training_data,
                feature_names=feature_names or self.config.feature_names,
                feature_ranges=self.config.feature_ranges
            )
            logger.info("✅ XAI system initialized with model and training data")
            return {
                "status": "success",
                "message": "XAI initialized",
                "features": len(feature_names) if feature_names else "unknown"
            }
        
        except Exception as e:
            logger.error(f"XAI initialization failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def explain_prediction(self, instance: np.ndarray,
                          methods: List[str] = None) -> Dict[str, Any]:
        """
        Get explanations for a prediction.
        
        Args:
            instance: Instance to explain
            methods: Explanation methods to use
            
        Returns:
            explanation: Multi-method explanations
        """
        if self.xai_controller is None:
            return {"status": "error", "message": "XAI not initialized"}
        
        explanations = self.xai_controller.explain_prediction(instance, methods)
        
        return {
            "status": "success",
            "explanations": {
                method: exp.to_dict()
                for method, exp in explanations.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_feature_importance(self, instance: np.ndarray) -> Dict[str, float]:
        """Get consensus feature importance."""
        if self.xai_controller is None:
            return {}
        
        return self.xai_controller.get_feature_importance_summary(instance)
    
    def get_feature_interactions(self, n_top: int = 10) -> List[Tuple]:
        """Get top feature interactions."""
        if self.xai_controller is None:
            return []
        
        return self.xai_controller.get_feature_interactions(n_top)
    
    # ========================================================================
    # Uncertainty Quantification Interface
    # ========================================================================
    
    def predict_with_uncertainty(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Get prediction with full uncertainty quantification.
        
        Args:
            instance: Instance to predict
            
        Returns:
            prediction: Prediction with uncertainty
        """
        if self.uncertainty_controller is None:
            return {"status": "error", "message": "Uncertainty quantification not enabled"}
        
        predictions = self.uncertainty_controller.predict_with_full_uncertainty(instance)
        ensemble = self.uncertainty_controller.ensemble_prediction(predictions)
        
        return {
            "status": "success",
            "prediction": ensemble.prediction,
            "confidence": ensemble.confidence,
            "total_uncertainty": ensemble.total_uncertainty,
            "aleatoric_uncertainty": ensemble.aleatoric_uncertainty,
            "epistemic_uncertainty": ensemble.epistemic_uncertainty,
            "is_ood": ensemble.is_ood,
            "is_reliable": ensemble.is_reliable(),
            "timestamp": datetime.now().isoformat()
        }
    
    # ========================================================================
    # Federated Learning Interface
    # ========================================================================
    
    def initialize_federated_learning(self, data_dict: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Initialize federated learning with client data.
        
        Args:
            data_dict: {client_id: (X, y)}
            
        Returns:
            status: Initialization status
        """
        if self.federated_learning is None:
            return {"status": "error", "message": "Federated learning not enabled"}
        
        try:
            self.federated_learning.initialize_clients(data_dict)
            return {
                "status": "success",
                "num_clients": len(data_dict),
                "message": "Federated learning initialized"
            }
        
        except Exception as e:
            logger.error(f"Federated learning initialization failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_federated_training(self, num_rounds: int,
                              client_fraction: float = 1.0,
                              local_epochs: int = 5) -> Dict[str, Any]:
        """
        Run federated training.
        
        Args:
            num_rounds: Number of federation rounds
            client_fraction: Fraction of clients per round
            local_epochs: Local training epochs
            
        Returns:
            result: Training results
        """
        if self.federated_learning is None:
            return {"status": "error", "message": "Federated learning not enabled"}
        
        logger.info(f"Starting federated training: {num_rounds} rounds")
        
        history = self.federated_learning.run_federated_training(
            num_rounds=num_rounds,
            client_fraction=client_fraction,
            local_epochs=local_epochs
        )
        
        self.federated_training_history.append({
            "timestamp": datetime.now().isoformat(),
            "num_rounds": num_rounds,
            "history_length": len(history)
        })
        
        return {
            "status": "success",
            "rounds_completed": len(history),
            "summary": self.federated_learning.get_training_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def share_threat_intelligence(self, attack_type: str,
                                 signature: str,
                                 confidence: float,
                                 affected_orgs: List[str],
                                 mitigations: List[str],
                                 organization: str) -> Dict[str, str]:
        """
        Share threat with federated network.
        
        Args:
            attack_type: Type of attack
            signature: Attack signature
            confidence: Detection confidence (0-1)
            affected_orgs: Organizations affected
            mitigations: Suggested mitigations
            organization: Submitting organization
            
        Returns:
            result: Sharing result
        """
        if self.federated_learning is None:
            return {"status": "error", "message": "Federated learning not enabled"}
        
        threat = ThreatIntelligence(
            attack_type=attack_type,
            signature=signature,
            confidence=confidence,
            affected_organizations=affected_orgs,
            mitigation_actions=mitigations
        )
        
        threat_id = self.federated_learning.share_threat_intelligence(threat, organization)
        
        self.threat_sharing_history.append({
            "threat_id": threat_id,
            "organization": organization,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Threat shared: {threat_id} from {organization}")
        
        return {
            "status": "success",
            "threat_id": threat_id,
            "shared_across": len(self.federated_learning.threat_hub.threat_repository)
        }
    
    def get_shared_threats(self, organization: str,
                          min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Get threats shared with organization."""
        if self.federated_learning is None:
            return []
        
        threats = self.federated_learning.get_shared_threats(organization)
        
        return [
            {
                "attack_type": t.attack_type,
                "signature": t.signature,
                "confidence": t.confidence,
                "mitigation_actions": t.mitigation_actions
            }
            for t in threats
            if t.confidence >= min_confidence
        ]
    
    # ========================================================================
    # System Status & Monitoring
    # ========================================================================
    
    def get_phase12_status(self) -> Dict[str, Any]:
        """Get comprehensive Phase 12 status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "meta_learning": {
                    "enabled": self.config.enable_meta_learning,
                    "initialized": self.meta_learning is not None,
                    "operations": (
                        self.meta_learning.get_performance_summary()
                        if self.meta_learning else {}
                    )
                },
                "xai": {
                    "enabled": self.config.enable_xai,
                    "initialized": self.xai_controller is not None
                },
                "uncertainty_quantification": {
                    "enabled": self.config.enable_uncertainty,
                    "initialized": self.uncertainty_controller is not None
                },
                "federated_learning": {
                    "enabled": self.config.enable_federated_learning,
                    "initialized": self.federated_learning is not None,
                    "summary": (
                        self.federated_learning.get_training_summary()
                        if self.federated_learning else {}
                    )
                }
            },
            "history": {
                "predictions": len(self.prediction_history),
                "threats_shared": len(self.threat_sharing_history),
                "federated_trainings": len(self.federated_training_history)
            }
        }
        
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "meta_learning": (
                self.meta_learning.get_performance_summary()
                if self.meta_learning else {}
            ),
            "threat_intelligence": (
                self.federated_learning.threat_hub.get_threat_coverage()
                if self.federated_learning else {}
            ),
            "prediction_history_size": len(self.prediction_history),
            "threat_sharing_history_size": len(self.threat_sharing_history)
        }
        
        return report


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'Phase12Configuration',
    'Phase12Integration'
]

if __name__ == "__main__":
    print("Phase 12 Integration Layer for AegisPCAP")
    print("Use: from api.phase12_integration import Phase12Integration")
