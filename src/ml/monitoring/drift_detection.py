"""Data and concept drift detection.

Monitors input data distribution and model performance to detect:
- Data drift: Input distribution changes
- Concept drift: Target distribution or decision boundary changes
- Prediction drift: Model output distribution changes

Classes:
    DataDriftDetector: Detects input data distribution changes
    ConceptDriftDetector: Detects decision boundary drift
    CalibrationMonitor: Tracks prediction calibration
    FeatureMonitor: Monitors individual feature statistics
"""

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class DriftType(Enum):
    """Types of drift."""
    NO_DRIFT = "no_drift"
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_DRIFT = "sudden_drift"
    RECURRING_DRIFT = "recurring_drift"


@dataclass
class DriftAlert:
    """Alert raised when drift is detected.
    
    Attributes:
        drift_type: Type of drift detected
        p_value: Statistical test p-value
        drift_magnitude: Magnitude of drift
        timestamp: When drift was detected
        description: Description of drift
    """
    drift_type: DriftType
    p_value: float
    drift_magnitude: float
    timestamp: datetime
    description: str = ""

    def __str__(self) -> str:
        """String representation."""
        return (f"[{self.drift_type.value.upper()}] "
                f"Magnitude: {self.drift_magnitude:.4f}, "
                f"p-value: {self.p_value:.4f}")


class DataDriftDetector:
    """Detects input data distribution changes.
    
    Uses statistical tests (Kolmogorov-Smirnov, ADWIN) to detect
    when feature distributions change significantly from baseline.
    
    Attributes:
        feature_name: Name of feature being monitored
        reference_data: Reference data for baseline
        window_size: Size of sliding window for drift detection
    """

    def __init__(
        self,
        feature_name: str,
        reference_data: np.ndarray,
        window_size: int = 100,
        alpha: float = 0.05,
    ):
        """Initialize data drift detector.
        
        Args:
            feature_name: Name of feature
            reference_data: Reference/baseline feature values
            window_size: Sliding window size for detection
            alpha: Significance level for statistical tests
        """
        self.feature_name = feature_name
        self.reference_data = reference_data
        self.window_size = window_size
        self.alpha = alpha
        
        self.data_window: Deque[float] = deque(maxlen=window_size)
        self.alerts: List[DriftAlert] = []
        
        # Compute reference statistics
        self.ref_mean = np.mean(reference_data)
        self.ref_std = np.std(reference_data)
        self.ref_quantiles = {
            q: np.quantile(reference_data, q)
            for q in [0.25, 0.5, 0.75]
        }

    def update(self, value: float) -> Optional[DriftAlert]:
        """Update detector with new value.
        
        Args:
            value: New feature value
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        self.data_window.append(value)
        
        # Need enough data for detection
        if len(self.data_window) < self.window_size // 2:
            return None
        
        # Perform statistical test
        window_data = np.array(self.data_window)
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(
            window_data,
            self.reference_data,
        )
        
        # Detect drift
        if p_value < self.alpha:
            drift_magnitude = float(ks_stat)
            drift_type = self._classify_drift_type(window_data)
            
            alert = DriftAlert(
                drift_type=drift_type,
                p_value=p_value,
                drift_magnitude=drift_magnitude,
                timestamp=datetime.now(),
                description=f"Data drift in {self.feature_name}",
            )
            self.alerts.append(alert)
            return alert
        
        return None

    def _classify_drift_type(self, window_data: np.ndarray) -> DriftType:
        """Classify type of drift.
        
        Args:
            window_data: Recent window of data
            
        Returns:
            DriftType classification
        """
        # Check for gradual drift (mean shift)
        window_mean = np.mean(window_data)
        mean_shift = abs(window_mean - self.ref_mean)
        
        # Check for sudden drift (large increase in variance)
        window_std = np.std(window_data)
        std_ratio = window_std / (self.ref_std + 1e-6)
        
        if mean_shift > 2 * self.ref_std:
            return DriftType.SUDDEN_DRIFT
        elif mean_shift > 0.5 * self.ref_std:
            return DriftType.GRADUAL_DRIFT
        elif std_ratio > 2.0:
            return DriftType.SUDDEN_DRIFT
        else:
            return DriftType.NO_DRIFT

    def get_stats(self) -> Dict[str, float]:
        """Get detector statistics.
        
        Returns:
            Dictionary of stats
        """
        if not self.data_window:
            return {}
        
        window_data = np.array(self.data_window)
        return {
            "window_mean": float(np.mean(window_data)),
            "window_std": float(np.std(window_data)),
            "reference_mean": float(self.ref_mean),
            "reference_std": float(self.ref_std),
            "mean_shift": float(abs(np.mean(window_data) - self.ref_mean)),
        }


class ConceptDriftDetector:
    """Detects decision boundary drift.
    
    Monitors model accuracy over time to detect when
    decision boundary changes (concept drift).
    
    Attributes:
        window_size: Size of sliding window for drift detection
        min_change: Minimum accuracy change to trigger drift
    """

    def __init__(
        self,
        window_size: int = 100,
        min_change: float = 0.05,
        alpha: float = 0.05,
    ):
        """Initialize concept drift detector.
        
        Args:
            window_size: Sliding window size
            min_change: Minimum accuracy change to trigger alert
            alpha: Significance level
        """
        self.window_size = window_size
        self.min_change = min_change
        self.alpha = alpha
        
        self.accuracy_window: Deque[float] = deque(maxlen=window_size)
        self.alerts: List[DriftAlert] = []
        self.baseline_accuracy = None

    def update(self, accuracy: float) -> Optional[DriftAlert]:
        """Update detector with new accuracy.
        
        Args:
            accuracy: Model accuracy (0-1)
            
        Returns:
            DriftAlert if drift detected
        """
        self.accuracy_window.append(accuracy)
        
        # Set baseline from first window
        if self.baseline_accuracy is None:
            if len(self.accuracy_window) >= self.window_size // 2:
                self.baseline_accuracy = np.mean(self.accuracy_window)
            return None
        
        # Check for significant accuracy drop
        window_accuracy = np.mean(self.accuracy_window)
        accuracy_change = self.baseline_accuracy - window_accuracy
        
        if accuracy_change > self.min_change:
            drift_type = DriftType.GRADUAL_DRIFT if accuracy_change < 0.1 else DriftType.SUDDEN_DRIFT
            
            alert = DriftAlert(
                drift_type=drift_type,
                p_value=0.0,
                drift_magnitude=float(accuracy_change),
                timestamp=datetime.now(),
                description=f"Concept drift: accuracy dropped by {accuracy_change:.4f}",
            )
            self.alerts.append(alert)
            self.baseline_accuracy = window_accuracy
            return alert
        
        return None


class CalibrationMonitor:
    """Monitors prediction calibration quality.
    
    Tracks Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    to detect when model confidence becomes miscalibrated.
    
    Attributes:
        n_bins: Number of bins for ECE calculation
    """

    def __init__(self, n_bins: int = 10):
        """Initialize calibration monitor.
        
        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.n_bins = n_bins
        self.predicted_probs: Deque[float] = deque(maxlen=1000)
        self.labels: Deque[int] = deque(maxlen=1000)

    def update(self, predicted_prob: float, label: int) -> None:
        """Update with prediction and label.
        
        Args:
            predicted_prob: Predicted probability (0-1)
            label: True label (0 or 1)
        """
        self.predicted_probs.append(predicted_prob)
        self.labels.append(label)

    def compute_ece(self) -> float:
        """Compute Expected Calibration Error.
        
        Returns:
            ECE value (0-1)
        """
        if len(self.predicted_probs) == 0:
            return 0.0
        
        probs = np.array(self.predicted_probs)
        labels = np.array(self.labels)
        
        ece = 0.0
        for i in range(self.n_bins):
            bin_lower = i / self.n_bins
            bin_upper = (i + 1) / self.n_bins
            
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            if not np.any(in_bin):
                continue
            
            bin_probs = probs[in_bin]
            bin_labels = labels[in_bin]
            
            bin_acc = np.mean(bin_labels)
            bin_conf = np.mean(bin_probs)
            
            ece += np.abs(bin_acc - bin_conf) * np.mean(in_bin)
        
        return float(ece)

    def compute_mce(self) -> float:
        """Compute Maximum Calibration Error.
        
        Returns:
            MCE value (0-1)
        """
        if len(self.predicted_probs) == 0:
            return 0.0
        
        probs = np.array(self.predicted_probs)
        labels = np.array(self.labels)
        
        mce = 0.0
        for i in range(self.n_bins):
            bin_lower = i / self.n_bins
            bin_upper = (i + 1) / self.n_bins
            
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            if not np.any(in_bin):
                continue
            
            bin_probs = probs[in_bin]
            bin_labels = labels[in_bin]
            
            bin_acc = np.mean(bin_labels)
            bin_conf = np.mean(bin_probs)
            
            mce = max(mce, abs(bin_acc - bin_conf))
        
        return float(mce)

    def get_calibration_curve(self) -> Tuple[List[float], List[float]]:
        """Get calibration curve (confidence vs accuracy).
        
        Returns:
            Tuple of (mean_confidences, accuracies) for each bin
        """
        if len(self.predicted_probs) == 0:
            return [], []
        
        probs = np.array(self.predicted_probs)
        labels = np.array(self.labels)
        
        confidences = []
        accuracies = []
        
        for i in range(self.n_bins):
            bin_lower = i / self.n_bins
            bin_upper = (i + 1) / self.n_bins
            
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            if not np.any(in_bin):
                continue
            
            bin_probs = probs[in_bin]
            bin_labels = labels[in_bin]
            
            confidences.append(float(np.mean(bin_probs)))
            accuracies.append(float(np.mean(bin_labels)))
        
        return confidences, accuracies


class FeatureMonitor:
    """Monitors individual feature statistics.
    
    Tracks mean, std, min, max for each feature to detect
    distribution changes.
    
    Attributes:
        feature_names: Names of features to monitor
        reference_stats: Reference statistics for each feature
    """

    def __init__(
        self,
        feature_names: List[str],
        reference_data: np.ndarray,
    ):
        """Initialize feature monitor.
        
        Args:
            feature_names: Names of features
            reference_data: Reference data (n_samples, n_features)
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # Compute reference statistics
        self.reference_stats = {}
        for i in range(self.n_features):
            feature_data = reference_data[:, i]
            self.reference_stats[feature_names[i]] = {
                "mean": float(np.mean(feature_data)),
                "std": float(np.std(feature_data)),
                "min": float(np.min(feature_data)),
                "max": float(np.max(feature_data)),
            }
        
        self.feature_history: Dict[str, Deque[float]] = {
            name: deque(maxlen=1000)
            for name in feature_names
        }

    def update(self, feature_values: np.ndarray) -> Dict[str, float]:
        """Update with new feature values.
        
        Args:
            feature_values: Feature values (n_features,)
            
        Returns:
            Dictionary of anomaly scores for each feature
        """
        anomalies = {}
        
        for i, name in enumerate(self.feature_names):
            value = feature_values[i]
            self.feature_history[name].append(value)
            
            # Z-score anomaly detection
            ref_stats = self.reference_stats[name]
            z_score = abs((value - ref_stats["mean"]) / (ref_stats["std"] + 1e-6))
            anomalies[name] = float(z_score)
        
        return anomalies

    def get_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """Get statistics for a feature.
        
        Args:
            feature_name: Name of feature
            
        Returns:
            Dictionary of statistics
        """
        if feature_name not in self.feature_history:
            return {}
        
        history = np.array(self.feature_history[feature_name])
        if len(history) == 0:
            return {}
        
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
        }
