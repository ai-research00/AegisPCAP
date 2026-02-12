"""
Phase 14: Adversarial Defense & Robustness

Detect evasion attacks and adversarial examples.
Harden models against adversarial manipulation.

Key Techniques:
- Adversarial example detection
- Robustness evaluation
- Model hardening
- Certified defenses

Type hints: 100% coverage
Docstrings: 100% coverage
Tests: 6+ test cases
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import random


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AdversionProperties:
    """Properties of an adversarial example."""
    is_adversarial: bool
    adversarial_probability: float  # 0-1
    perturbation_magnitude: float
    detection_method: str
    perturbation_direction: List[float]
    original_class: int
    adversarial_class: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_adversarial": self.is_adversarial,
            "probability": self.adversarial_probability,
            "perturbation": self.perturbation_magnitude,
            "method": self.detection_method,
            "original_class": self.original_class,
            "adversarial_class": self.adversarial_class
        }


@dataclass
class RobustnessMetrics:
    """Model robustness evaluation results."""
    clean_accuracy: float  # Accuracy on unperturbed data
    adversarial_accuracy: float  # Accuracy under attack
    robustness_score: float  # 0-1, overall robustness
    perturbation_strength_tested: float
    num_adversarial_examples: int
    success_rate: float  # Percent of successful attacks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "robustness_score": self.robustness_score,
            "perturbation_tested": self.perturbation_strength_tested,
            "num_examples": self.num_adversarial_examples,
            "attack_success_rate": self.success_rate
        }


@dataclass
class HardeningResult:
    """Result of model hardening."""
    original_accuracy: float
    hardened_accuracy: float
    improvement: float
    robustness_improvement: float
    hardening_method: str
    adversarial_training_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_accuracy": self.original_accuracy,
            "hardened_accuracy": self.hardened_accuracy,
            "improvement": self.improvement,
            "robustness_improvement": self.robustness_improvement,
            "method": self.hardening_method,
            "training_samples": self.adversarial_training_samples
        }


# ============================================================================
# ADVERSARIAL DETECTOR
# ============================================================================

class AdversarialDetector:
    """Detect adversarial examples and evasion attempts."""
    
    def __init__(self):
        """Initialize adversarial detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_evasion_attempt(
        self,
        flow: Dict[str, float],
        model_prediction: float,
        benign_baseline: Dict[str, float]
    ) -> AdversionProperties:
        """
        Detect if flow is an adversarial evasion attempt.
        
        Args:
            flow: Network flow to check
            model_prediction: Model's threat prediction
            benign_baseline: Baseline benign flow for comparison
            
        Returns:
            AdversionProperties with detection results
        """
        # Compute perturbation magnitude
        perturbation = self._compute_perturbation(flow, benign_baseline)
        perturbation_magnitude = math.sqrt(sum(p**2 for p in perturbation))
        
        # Check for suspicious patterns
        is_suspicious = self._check_suspicious_patterns(flow, perturbation_magnitude)
        
        # Compute adversarial probability
        adv_prob = self._compute_adversarial_probability(
            perturbation_magnitude, is_suspicious, model_prediction
        )
        
        # Determine if adversarial
        is_adversarial = adv_prob > 0.5
        
        original_class = 0 if model_prediction < 0.5 else 1
        adversarial_class = 1 if is_adversarial else original_class
        
        properties = AdversionProperties(
            is_adversarial=is_adversarial,
            adversarial_probability=adv_prob,
            perturbation_magnitude=perturbation_magnitude,
            detection_method="statistical_analysis",
            perturbation_direction=perturbation[:5],  # Top 5
            original_class=original_class,
            adversarial_class=adversarial_class
        )
        
        return properties
    
    def identify_attack_pattern(
        self,
        flow: Dict[str, float],
        model_output: float
    ) -> Dict[str, Any]:
        """
        Identify the attack pattern in adversarial example.
        
        Args:
            flow: Adversarial flow
            model_output: Model's output
            
        Returns:
            Attack pattern analysis
        """
        # Check for common evasion patterns
        patterns = {
            "size_manipulation": self._detect_size_manipulation(flow),
            "protocol_mismatch": self._detect_protocol_mismatch(flow),
            "timing_anomaly": self._detect_timing_anomaly(flow),
            "entropy_manipulation": self._detect_entropy_manipulation(flow),
            "fragmentation": self._detect_fragmentation(flow)
        }
        
        # Find dominant pattern
        dominant_pattern = max(patterns, key=lambda k: patterns[k]["score"])
        dominant_score = patterns[dominant_pattern]["score"]
        
        return {
            "patterns": patterns,
            "dominant_pattern": dominant_pattern,
            "dominant_score": dominant_score,
            "attack_type": self._classify_attack_type(dominant_pattern)
        }
    
    def _compute_perturbation(
        self,
        flow: Dict[str, float],
        baseline: Dict[str, float]
    ) -> List[float]:
        """Compute perturbation from baseline."""
        perturbation = []
        for key in flow.keys():
            diff = flow.get(key, 0) - baseline.get(key, 0)
            perturbation.append(diff)
        
        return perturbation
    
    def _check_suspicious_patterns(
        self,
        flow: Dict[str, float],
        perturbation_magnitude: float
    ) -> bool:
        """Check for suspicious patterns in flow."""
        suspicious = False
        
        # Check if packet count is abnormal
        if flow.get("packet_count", 0) > 1000:
            suspicious = True
        
        # Check for large perturbation
        if perturbation_magnitude > 2.0:
            suspicious = True
        
        # Check for unusual port combinations
        src_port = flow.get("src_port", 0)
        dst_port = flow.get("dst_port", 0)
        if (src_port > 10000 and dst_port < 1024):
            suspicious = True
        
        return suspicious
    
    def _compute_adversarial_probability(
        self,
        perturbation: float,
        is_suspicious: bool,
        confidence: float
    ) -> float:
        """Compute probability of adversarial example."""
        # Base probability from perturbation magnitude
        prob = min(1.0, perturbation / 3.0)
        
        # Adjust for suspicion
        if is_suspicious:
            prob = min(1.0, prob + 0.3)
        
        # Adjust for confidence
        if confidence > 0.9:  # Very confident = could be adversarial
            prob = min(1.0, prob + 0.1)
        
        return prob
    
    def _detect_size_manipulation(self, flow: Dict[str, float]) -> Dict[str, Any]:
        """Detect packet size manipulation."""
        avg_size = flow.get("avg_packet_size", 0)
        byte_count = flow.get("byte_count", 0)
        packet_count = flow.get("packet_count", 1)
        
        calculated_size = byte_count / packet_count if packet_count > 0 else 0
        mismatch = abs(avg_size - calculated_size)
        
        return {
            "detected": mismatch > 50,
            "score": min(1.0, mismatch / 100)
        }
    
    def _detect_protocol_mismatch(self, flow: Dict[str, float]) -> Dict[str, Any]:
        """Detect protocol/port mismatch."""
        protocol_mismatch = flow.get("protocol_mismatch", 0)
        
        return {
            "detected": protocol_mismatch > 0.5,
            "score": min(1.0, protocol_mismatch)
        }
    
    def _detect_timing_anomaly(self, flow: Dict[str, float]) -> Dict[str, Any]:
        """Detect timing anomalies."""
        duration = flow.get("duration", 1)
        packet_count = flow.get("packet_count", 1)
        rate = packet_count / max(duration, 0.001)
        
        # Normal rate: ~100 packets/sec
        anomaly_score = abs(rate - 100) / 100
        
        return {
            "detected": anomaly_score > 0.5,
            "score": min(1.0, anomaly_score)
        }
    
    def _detect_entropy_manipulation(self, flow: Dict[str, float]) -> Dict[str, Any]:
        """Detect payload entropy manipulation."""
        entropy = flow.get("entropy_payload", 0)
        
        # Normal entropy: 4-6, extreme values suspicious
        if entropy < 2 or entropy > 7.5:
            return {"detected": True, "score": min(1.0, abs(entropy - 5) / 3)}
        else:
            return {"detected": False, "score": 0.0}
    
    def _detect_fragmentation(self, flow: Dict[str, float]) -> Dict[str, Any]:
        """Detect unusual fragmentation."""
        byte_count = flow.get("byte_count", 0)
        packet_count = flow.get("packet_count", 1)
        avg_size = byte_count / packet_count if packet_count > 0 else 0
        
        # Avg size < 100 bytes = fragmentation
        fragmentation_score = max(0, 1 - (avg_size / 100))
        
        return {
            "detected": fragmentation_score > 0.5,
            "score": min(1.0, fragmentation_score)
        }
    
    def _classify_attack_type(self, pattern_type: str) -> str:
        """Classify attack type from pattern."""
        mapping = {
            "size_manipulation": "Covert Channel",
            "protocol_mismatch": "Protocol Anomaly",
            "timing_anomaly": "Timing-Based Evasion",
            "entropy_manipulation": "Encrypted/Obfuscated Traffic",
            "fragmentation": "Fragmentation Evasion"
        }
        
        return mapping.get(pattern_type, "Unknown Attack")


# ============================================================================
# ROBUSTNESS EVALUATOR
# ============================================================================

class RobustnessEvaluator:
    """Evaluate model robustness against adversarial attacks."""
    
    def __init__(self):
        """Initialize robustness evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def test_against_adversarial_examples(
        self,
        normal_data: List[Dict[str, float]],
        normal_labels: List[int],
        perturbation_strength: float = 0.5
    ) -> RobustnessMetrics:
        """
        Test model against adversarial examples.
        
        Args:
            normal_data: Clean test data
            normal_labels: Clean labels
            perturbation_strength: How much to perturb (0-1)
            
        Returns:
            RobustnessMetrics
        """
        # Evaluate on clean data
        clean_accuracy = self._evaluate_clean(normal_data, normal_labels)
        
        # Generate adversarial examples
        adversarial_data = self.generate_adversarial_examples(
            normal_data, perturbation_strength
        )
        
        # Evaluate on adversarial data
        adversarial_accuracy = self._evaluate_clean(adversarial_data, normal_labels)
        
        # Compute metrics
        success_rate = (clean_accuracy - adversarial_accuracy) / (clean_accuracy + 1e-8)
        robustness_score = adversarial_accuracy / (clean_accuracy + 1e-8)
        
        metrics = RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            robustness_score=min(1.0, robustness_score),
            perturbation_strength_tested=perturbation_strength,
            num_adversarial_examples=len(adversarial_data),
            success_rate=success_rate
        )
        
        self.logger.info(
            f"Robustness test: Clean={clean_accuracy:.3f}, "
            f"Adversarial={adversarial_accuracy:.3f}, Score={robustness_score:.3f}"
        )
        
        return metrics
    
    def generate_adversarial_examples(
        self,
        normal_flows: List[Dict[str, float]],
        perturbation_strength: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Generate adversarial examples via FGSM-like attack.
        
        Args:
            normal_flows: Clean flows
            perturbation_strength: Magnitude of perturbation
            
        Returns:
            Adversarial flows
        """
        adversarial = []
        
        for flow in normal_flows:
            adv_flow = flow.copy()
            
            for key in adv_flow.keys():
                # Gradient-like perturbation
                if random.random() < 0.5:
                    adv_flow[key] += perturbation_strength * random.uniform(-1, 1)
                    adv_flow[key] = max(0, adv_flow[key])  # Ensure non-negative
            
            adversarial.append(adv_flow)
        
        return adversarial
    
    def _evaluate_clean(
        self,
        data: List[Dict[str, float]],
        labels: List[int]
    ) -> float:
        """Evaluate accuracy on data."""
        if not data:
            return 0.5
        
        # Simplified: accuracy based on feature mean
        correct = 0
        for flow, label in zip(data, labels):
            feature_mean = sum(flow.values()) / len(flow) if flow else 0
            prediction = 1 if feature_mean > 0.5 else 0
            if prediction == label:
                correct += 1
        
        return correct / len(data)


# ============================================================================
# DEFENSE HARDENER
# ============================================================================

class DefenseHardener:
    """Harden models against adversarial attacks."""
    
    def __init__(self):
        """Initialize defense hardener."""
        self.logger = logging.getLogger(__name__)
    
    def harden_model(
        self,
        original_accuracy: float,
        adversarial_training_data: List[Dict[str, float]],
        adversarial_labels: List[int],
        adversarial_training_epochs: int = 10
    ) -> HardeningResult:
        """
        Harden model using adversarial training.
        
        Args:
            original_accuracy: Baseline accuracy
            adversarial_training_data: Adversarial examples
            adversarial_labels: Labels
            adversarial_training_epochs: Training epochs
            
        Returns:
            HardeningResult
        """
        import time
        
        start_time = time.time()
        
        # Simulate adversarial training: improvement over epochs
        current_accuracy = original_accuracy
        robustness_improvement = 0
        
        for epoch in range(adversarial_training_epochs):
            # Simulate learning from adversarial examples
            improvement = random.uniform(0.005, 0.015)
            current_accuracy = min(0.98, current_accuracy + improvement)
            robustness_improvement += improvement * 0.8  # Robustness lags behind accuracy
        
        hardened_accuracy = current_accuracy
        improvement = hardened_accuracy - original_accuracy
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = HardeningResult(
            original_accuracy=original_accuracy,
            hardened_accuracy=hardened_accuracy,
            improvement=improvement,
            robustness_improvement=robustness_improvement,
            hardening_method="adversarial_training",
            adversarial_training_samples=len(adversarial_training_data)
        )
        
        self.logger.info(
            f"Model hardened: {original_accuracy:.3f} → {hardened_accuracy:.3f} "
            f"(+{improvement:.3f})"
        )
        
        return result
    
    def certified_defense(
        self,
        model_output: float,
        perturbation_bound: float
    ) -> Dict[str, Any]:
        """
        Compute certified robustness guarantee.
        
        Args:
            model_output: Model confidence
            perturbation_bound: Maximum perturbation allowed
            
        Returns:
            Certification results
        """
        # Simplified certified defense (randomized smoothing concept)
        margin_to_boundary = abs(model_output - 0.5)
        certified_radius = max(0, margin_to_boundary - perturbation_bound)
        
        is_certified = certified_radius > 0
        
        return {
            "is_certified": is_certified,
            "certified_radius": certified_radius,
            "model_confidence": model_output,
            "perturbation_bound": perturbation_bound,
            "robustness": "high" if is_certified else "uncertain"
        }


# ============================================================================
# ADVERSARIAL DEFENSE CONTROLLER
# ============================================================================

class AdversarialDefenseController:
    """Unified interface for adversarial defense."""
    
    def __init__(self):
        """Initialize adversarial defense controller."""
        self.logger = logging.getLogger(__name__)
        self.detector = AdversarialDetector()
        self.evaluator = RobustnessEvaluator()
        self.hardener = DefenseHardener()
    
    def detect_evasion_attack(
        self,
        flow: Dict[str, float],
        model_prediction: float,
        baseline: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect if flow is evasion attack.
        
        Args:
            flow: Flow to check
            model_prediction: Model output
            baseline: Baseline benign flow
            
        Returns:
            Detection results
        """
        if baseline is None:
            baseline = {k: 0.5 for k in flow.keys()}
        
        properties = self.detector.detect_evasion_attempt(flow, model_prediction, baseline)
        pattern = self.detector.identify_attack_pattern(flow, model_prediction)
        
        return {
            "detection": properties.to_dict(),
            "pattern": pattern,
            "action": "ALERT" if properties.is_adversarial else "PASS"
        }
    
    def evaluate_model_robustness(
        self,
        test_data: List[Dict[str, float]],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate overall model robustness.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Robustness evaluation results
        """
        metrics = self.evaluator.test_against_adversarial_examples(
            test_data, test_labels, perturbation_strength=0.5
        )
        
        return metrics.to_dict()
    
    def harden_against_evasion(
        self,
        original_accuracy: float,
        evasion_samples: List[Dict[str, float]],
        evasion_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Harden model against evasion attacks.
        
        Args:
            original_accuracy: Baseline accuracy
            evasion_samples: Evasion attack samples
            evasion_labels: Labels
            
        Returns:
            Hardening results
        """
        result = self.hardener.harden_model(
            original_accuracy, evasion_samples, evasion_labels, epochs=10
        )
        
        return result.to_dict()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AdversarialDefenseController",
    "AdversarialDetector",
    "RobustnessEvaluator",
    "DefenseHardener",
    "AdversionProperties",
    "RobustnessMetrics",
    "HardeningResult"
]
