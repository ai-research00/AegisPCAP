"""
Uncertainty Quantification for Phase 10: Advanced Analytics

Implements:
- Bayesian Neural Networks for epistemic uncertainty
- Monte Carlo Dropout for aleatoric and epistemic uncertainty
- Ensemble-based uncertainty estimation
- Calibration methods for reliable confidence estimates
- Conformal prediction for distribution-free uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ===== DATA STRUCTURES =====

@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for predictions."""
    
    predictions: torch.Tensor  # (B, num_classes) or (B,)
    aleatoric_uncertainty: torch.Tensor  # Measurement noise
    epistemic_uncertainty: torch.Tensor  # Model uncertainty
    total_uncertainty: torch.Tensor  # Sum of above
    confidence: torch.Tensor  # 1 - max uncertainty


# ===== BAYESIAN NEURAL NETWORKS =====

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Implements variational inference for weight distributions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_scale: float = 1.0
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            prior_scale: Scale of weight prior (standard deviation)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale
        
        # Weight mean and log variance (variational parameters)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) / in_features)
        self.weight_log_var = nn.Parameter(torch.ones(out_features, in_features) * -5.0)
        
        # Bias mean and log variance
        self.bias_mean = nn.Parameter(torch.randn(out_features) / in_features)
        self.bias_log_var = nn.Parameter(torch.ones(out_features) * -5.0)
        
        # KL divergence accumulator
        self.register_buffer('kl_divergence', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor (B, in_features)
        
        Returns:
            Output (B, out_features)
        """
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight_eps = torch.randn_like(weight_std)
        weights = self.weight_mean + weight_eps * weight_std
        
        # Sample biases
        bias_std = torch.exp(0.5 * self.bias_log_var)
        bias_eps = torch.randn_like(bias_std)
        biases = self.bias_mean + bias_eps * bias_std
        
        # Compute KL divergence
        prior_var = self.prior_scale ** 2
        self.kl_divergence = self._compute_kl_divergence(
            self.weight_mean, self.weight_log_var,
            self.bias_mean, self.bias_log_var,
            prior_var
        )
        
        return F.linear(x, weights, biases)
    
    @staticmethod
    def _compute_kl_divergence(
        weight_mean: torch.Tensor,
        weight_log_var: torch.Tensor,
        bias_mean: torch.Tensor,
        bias_log_var: torch.Tensor,
        prior_var: float
    ) -> torch.Tensor:
        """Compute KL divergence for variational weights."""
        # Weight KL
        weight_var = torch.exp(weight_log_var)
        weight_kl = 0.5 * torch.sum(
            (weight_var + weight_mean ** 2) / prior_var
            - 1.0 - weight_log_var + np.log(prior_var)
        )
        
        # Bias KL
        bias_var = torch.exp(bias_log_var)
        bias_kl = 0.5 * torch.sum(
            (bias_var + bias_mean ** 2) / prior_var
            - 1.0 - bias_log_var + np.log(prior_var)
        )
        
        return weight_kl + bias_kl


class BayesianNetwork(nn.Module):
    """
    Bayesian neural network for uncertainty quantification.
    
    Uses variational inference for weight uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_scale: float = 1.0,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Bayesian network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            prior_scale: Prior weight scale
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i+1], prior_scale))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with KL divergence.
        
        Args:
            x: Input tensor (B, input_dim)
        
        Returns:
            Tuple of (output logits, total KL divergence)
        """
        total_kl = 0
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            total_kl = total_kl + layer.kl_divergence
            
            # ReLU and dropout for all but last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x, total_kl
    
    def sample_predictions(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Sample multiple predictions for uncertainty.
        
        Args:
            x: Input tensor (B, input_dim)
            num_samples: Number of MC samples
        
        Returns:
            Sample predictions (num_samples, B, output_dim)
        """
        predictions = []
        
        for _ in range(num_samples):
            logits, _ = self.forward(x)
            predictions.append(logits)
        
        return torch.stack(predictions)


# ===== MONTE CARLO DROPOUT =====

class MCDropout(nn.Module):
    """
    Applies dropout at test time for uncertainty estimation.
    
    Provides both aleatoric (data) and epistemic (model) uncertainty.
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize MC Dropout.
        
        Args:
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout even during inference."""
        return F.dropout(x, p=self.dropout_rate, training=True)


class MCDropoutNetwork(nn.Module):
    """
    Neural network with MC Dropout for uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.5
    ):
        """Initialize MC Dropout network."""
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:  # No dropout on output layer
                self.layers.append(MCDropout(dropout_rate))
                self.layers.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, input_dim)
            return_all: Whether to keep dropout in eval mode
        
        Returns:
            Output logits (B, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def sample_predictions(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Sample predictions using MC Dropout.
        
        Args:
            x: Input tensor (B, input_dim)
            num_samples: Number of stochastic forward passes
        
        Returns:
            Sample predictions (num_samples, B, output_dim)
        """
        was_training = self.training
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                predictions.append(logits)
        
        if not was_training:
            self.eval()
        
        return torch.stack(predictions)


# ===== ENSEMBLE UNCERTAINTY =====

class EnsembleUncertaintyEstimator:
    """
    Estimates uncertainty from ensemble predictions.
    
    Computes epistemic uncertainty from ensemble disagreement.
    """
    
    def __init__(self, models: List[nn.Module], device: torch.device):
        """
        Initialize ensemble estimator.
        
        Args:
            models: List of trained models
            device: Device for computation
        """
        self.models = [m.to(device).eval() for m in models]
        self.device = device
    
    def estimate_uncertainty(
        self,
        x: torch.Tensor
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from ensemble.
        
        Args:
            x: Input tensor (B, input_dim)
        
        Returns:
            UncertaintyEstimate with epistemic and aleatoric components
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (num_models, B, num_classes)
        
        # Mean prediction
        mean_pred = torch.mean(predictions, dim=0)  # (B, num_classes)
        
        # Epistemic uncertainty (model disagreement)
        epistemic = torch.var(predictions, dim=0)  # (B, num_classes)
        epistemic = torch.mean(epistemic, dim=1)   # Average across classes
        
        # Aleatoric uncertainty (max probability entropy)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
        aleatoric = entropy / np.log(mean_pred.shape[1])  # Normalize by num classes
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        # Confidence
        confidence = 1.0 - torch.max(total, dim=-1 if total.ndim > 1 else 0)[0]
        if confidence.ndim == 0:
            confidence = torch.mean(total, dim=-1) if total.ndim > 1 else total
            confidence = 1.0 - confidence
        
        return UncertaintyEstimate(
            predictions=mean_pred,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            total_uncertainty=total,
            confidence=confidence
        )


# ===== CALIBRATION =====

class TemperatureScaling:
    """
    Temperature scaling for calibrating confidence estimates.
    
    Adjusts softmax temperature to improve uncertainty calibration.
    """
    
    def __init__(self):
        """Initialize temperature scaler."""
        self.temperature = 1.0
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = 0.01,
        max_iterations: int = 100
    ):
        """
        Fit temperature parameter.
        
        Args:
            logits: Model logits from validation set (N, num_classes)
            labels: Ground truth labels (N,)
            learning_rate: Optimization learning rate
            max_iterations: Maximum optimization steps
        """
        self.temperature = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        optimizer = torch.optim.LBFGS([self.temperature], lr=learning_rate)
        
        def closure():
            optimizer.zero_grad()
            
            # Compute loss with current temperature
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            
            loss.backward()
            return loss
        
        for _ in range(max_iterations):
            optimizer.step(closure)
        
        self.temperature = self.temperature.item()
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.
        
        Args:
            logits: Model logits (B, num_classes)
        
        Returns:
            Calibrated logits (B, num_classes)
        """
        return logits / self.temperature


class IsotonicCalibration:
    """
    Isotonic regression for calibration.
    
    Maps confidence to accuracy using isotonic regression.
    """
    
    def __init__(self):
        """Initialize isotonic calibration."""
        self.bins = 10
        self.bin_boundaries = np.linspace(0, 1, self.bins + 1)
        self.calibration_map = None
    
    def fit(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ):
        """
        Fit isotonic calibration.
        
        Args:
            confidences: Model confidences (N,)
            accuracies: Binary correctness labels (N,)
        """
        # Bin confidences
        binned = np.digitize(confidences, self.bin_boundaries) - 1
        binned = np.clip(binned, 0, self.bins - 1)
        
        # Compute accuracy per bin
        self.calibration_map = {}
        for b in range(self.bins):
            mask = binned == b
            if mask.sum() > 0:
                self.calibration_map[b] = accuracies[mask].mean()
            else:
                self.calibration_map[b] = 0.5
    
    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            confidences: Model confidences (N,)
        
        Returns:
            Calibrated confidences (N,)
        """
        if self.calibration_map is None:
            return confidences
        
        binned = np.digitize(confidences, self.bin_boundaries) - 1
        binned = np.clip(binned, 0, self.bins - 1)
        
        calibrated = np.array([
            self.calibration_map.get(b, 0.5)
            for b in binned
        ])
        
        return calibrated


# ===== CONFORMAL PREDICTION =====

class ConformalPredictor:
    """
    Conformal prediction for distribution-free uncertainty sets.
    
    Creates prediction sets with guaranteed coverage probability.
    """
    
    def __init__(self, significance_level: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            significance_level: 1 - target coverage probability
        """
        self.significance_level = significance_level
        self.threshold = None
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Fit nonconformity threshold on calibration set.
        
        Args:
            logits: Validation logits (N, num_classes)
            labels: Validation labels (N,)
        """
        # Compute nonconformity scores
        probs = F.softmax(logits, dim=1)
        correct_probs = probs[torch.arange(len(labels)), labels]
        
        # Threshold at (1 - alpha) quantile
        quantile_idx = int(np.ceil((len(labels) + 1) * (1 - self.significance_level)))
        quantile_idx = min(quantile_idx, len(labels) - 1)
        
        sorted_scores = torch.sort(correct_probs)[0]
        self.threshold = sorted_scores[quantile_idx].item()
    
    def predict_set(
        self,
        logits: torch.Tensor
    ) -> List[List[int]]:
        """
        Create prediction sets.
        
        Args:
            logits: Test logits (B, num_classes)
        
        Returns:
            List of prediction sets (list of class indices per sample)
        """
        if self.threshold is None:
            raise RuntimeError("Must fit before making predictions")
        
        probs = F.softmax(logits, dim=1)
        
        prediction_sets = []
        for i in range(len(probs)):
            # Classes with prob above threshold
            confident_classes = torch.where(probs[i] >= self.threshold)[0].tolist()
            
            # If empty, include highest probability class
            if not confident_classes:
                confident_classes = [torch.argmax(probs[i]).item()]
            
            prediction_sets.append(confident_classes)
        
        return prediction_sets


# ===== UTILITY FUNCTIONS =====

def compute_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Compute expected calibration error (ECE).
    
    Measures difference between confidence and accuracy across bins.
    
    Args:
        confidences: Model confidence scores (N,)
        accuracies: Binary correctness labels (N,)
        num_bins: Number of confidence bins
    
    Returns:
        Expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    binned = np.digitize(confidences, bin_boundaries) - 1
    
    ece = 0
    for b in range(num_bins):
        mask = binned == b
        if mask.sum() == 0:
            continue
        
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        
        ece += (mask.sum() / len(confidences)) * abs(bin_conf - bin_acc)
    
    return ece


def compute_uncertainty_coverage(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    labels: np.ndarray,
    confidence_levels: List[float] = None
) -> Dict[str, float]:
    """
    Compute coverage-uncertainty tradeoff.
    
    Args:
        predictions: Model predictions (N,)
        uncertainties: Uncertainty estimates (N,)
        labels: Ground truth labels (N,)
        confidence_levels: List of confidence thresholds
    
    Returns:
        Dictionary of coverage and accuracy at each level
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.7, 0.9, 0.95]
    
    results = {}
    
    for conf_level in confidence_levels:
        mask = uncertainties <= (1 - conf_level)
        
        if mask.sum() == 0:
            results[f'coverage_{conf_level}'] = 0.0
            results[f'accuracy_{conf_level}'] = 0.0
        else:
            coverage = mask.sum() / len(mask)
            accuracy = (predictions[mask] == labels[mask]).mean()
            
            results[f'coverage_{conf_level}'] = float(coverage)
            results[f'accuracy_{conf_level}'] = float(accuracy)
    
    return results


if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Bayesian network
    bay_net = BayesianNetwork(10, [32, 32], 5).to(device)
    
    # Sample predictions
    x = torch.randn(4, 10).to(device)
    samples = bay_net.sample_predictions(x, num_samples=5)
    print(f"Bayesian samples shape: {samples.shape}")
    
    # MC Dropout
    mc_net = MCDropoutNetwork(10, [32, 32], 5).to(device)
    mc_samples = mc_net.sample_predictions(x, num_samples=5)
    print(f"MC Dropout samples shape: {mc_samples.shape}")
    
    # Ensemble uncertainty
    models = [MCDropoutNetwork(10, [32, 32], 5).to(device) for _ in range(3)]
    estimator = EnsembleUncertaintyEstimator(models, device)
    uncertainty = estimator.estimate_uncertainty(x)
    print(f"Uncertainty estimate shape: {uncertainty.total_uncertainty.shape}")
