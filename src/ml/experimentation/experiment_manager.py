"""Experiment management and A/B testing framework.

Manages ML experiments including model variants, tracking,
versioning, and deployment strategies.

Classes:
    ModelVariant: Represents a model variant in experiment
    ExperimentConfig: Configuration for A/B test
    ExperimentManager: Manages experiments end-to-end
    ExperimentTracker: Tracks experiment metrics
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExperimentStatus(Enum):
    """Experiment status."""
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelVariant:
    """Model variant in experiment.
    
    Attributes:
        name: Variant name (e.g., "control", "treatment_v1")
        model_id: Unique model identifier
        version: Model version
        traffic_percentage: Traffic allocation (0-100)
        description: Variant description
    """
    name: str
    model_id: str
    version: str
    traffic_percentage: float = 50.0
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate variant."""
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("traffic_percentage must be 0-100")


@dataclass
class ExperimentConfig:
    """A/B test configuration.
    
    Attributes:
        name: Experiment name
        description: Experiment description
        variants: List of model variants
        metrics: Metrics to track
        sample_size: Target sample size
        duration_days: Experiment duration in days
        significance_level: Statistical significance level (alpha)
        min_relative_lift: Minimum relative lift to detect
    """
    name: str
    description: str
    variants: List[ModelVariant]
    metrics: List[str] = field(default_factory=list)
    sample_size: int = 10000
    duration_days: int = 7
    significance_level: float = 0.05
    min_relative_lift: float = 0.05
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration invalid
        """
        if len(self.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_traffic}%")
        
        if self.sample_size < 100:
            raise ValueError("Sample size must be at least 100")
        
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        
        return True


class ExperimentTracker:
    """Tracks experiment metrics and results.
    
    Attributes:
        experiment_name: Name of experiment
        variant_metrics: Metrics for each variant
    """

    def __init__(self, experiment_name: str, variant_names: List[str]):
        """Initialize tracker.
        
        Args:
            experiment_name: Name of experiment
            variant_names: Names of variants being tracked
        """
        self.experiment_name = experiment_name
        self.variant_names = variant_names
        self.variant_metrics: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in variant_names
        }
        self.start_time = time.time()

    def record(
        self,
        variant_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Record metrics for a variant.
        
        Args:
            variant_name: Name of variant
            metrics: Dictionary of metric values
        """
        if variant_name not in self.variant_metrics:
            raise ValueError(f"Unknown variant: {variant_name}")
        
        metrics["timestamp"] = time.time()
        self.variant_metrics[variant_name].append(metrics)

    def get_variant_stats(self, variant_name: str) -> Dict[str, float]:
        """Get aggregated statistics for variant.
        
        Args:
            variant_name: Name of variant
            
        Returns:
            Dictionary of statistics
        """
        if not self.variant_metrics[variant_name]:
            return {}
        
        records = self.variant_metrics[variant_name]
        
        # Extract metric columns (exclude timestamp)
        metric_names = [k for k in records[0].keys() if k != "timestamp"]
        stats = {}
        
        for metric_name in metric_names:
            values = [r[metric_name] for r in records]
            values = np.array(values)
            
            stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "n": len(values),
            }
        
        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all variants.
        
        Returns:
            Dictionary mapping variant names to stats
        """
        return {
            name: self.get_variant_stats(name)
            for name in self.variant_names
        }

    def get_sample_size(self, variant_name: str) -> int:
        """Get number of samples for variant.
        
        Args:
            variant_name: Name of variant
            
        Returns:
            Number of samples
        """
        return len(self.variant_metrics[variant_name])

    def get_total_samples(self) -> int:
        """Get total samples across all variants.
        
        Returns:
            Total sample count
        """
        return sum(len(records) for records in self.variant_metrics.values())

    def get_duration_seconds(self) -> float:
        """Get experiment duration in seconds.
        
        Returns:
            Duration in seconds
        """
        return time.time() - self.start_time


class ExperimentManager:
    """Manages ML experiments and A/B tests.
    
    Orchestrates experiment lifecycle including configuration,
    execution, analysis, and deployment decisions.
    
    Attributes:
        experiment_name: Name of experiment
        config: Experiment configuration
        status: Current experiment status
        tracker: Metrics tracker
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment manager.
        
        Args:
            config: Experiment configuration
        """
        config.validate()
        
        self.config = config
        self.status = ExperimentStatus.PLANNING
        self.variant_names = [v.name for v in config.variants]
        self.tracker = ExperimentTracker(config.name, self.variant_names)
        self.results: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        """Start experiment."""
        if self.status != ExperimentStatus.PLANNING:
            raise ValueError(f"Cannot start experiment in {self.status} state")
        
        self.status = ExperimentStatus.RUNNING
        print(f"Started experiment: {self.config.name}")

    def pause(self) -> None:
        """Pause experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in {self.status} state")
        
        self.status = ExperimentStatus.PAUSED

    def resume(self) -> None:
        """Resume experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot resume experiment in {self.status} state")
        
        self.status = ExperimentStatus.RUNNING

    def record_metrics(
        self,
        variant_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Record metrics for a variant.
        
        Args:
            variant_name: Name of variant
            metrics: Dictionary of metric values
        """
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot record metrics in {self.status} state")
        
        self.tracker.record(variant_name, metrics)

    def is_sample_size_reached(self) -> bool:
        """Check if target sample size reached.
        
        Returns:
            True if sample size reached
        """
        return self.tracker.get_total_samples() >= self.config.sample_size

    def is_duration_exceeded(self, start_time: float) -> bool:
        """Check if experiment duration exceeded.
        
        Args:
            start_time: Experiment start time (unix timestamp)
            
        Returns:
            True if duration exceeded
        """
        duration_seconds = time.time() - start_time
        target_seconds = self.config.duration_days * 86400
        return duration_seconds > target_seconds

    def complete(self) -> None:
        """Complete experiment and analyze results."""
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError(f"Cannot complete experiment in {self.status} state")
        
        self.status = ExperimentStatus.COMPLETED
        self.results = self._analyze_results()

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Returns:
            Dictionary of analysis results
        """
        stats = self.tracker.get_all_stats()
        
        # Get variant objects for reference
        variant_map = {v.name: v for v in self.config.variants}
        
        results = {
            "experiment_name": self.config.name,
            "status": self.status.value,
            "duration_seconds": self.tracker.get_duration_seconds(),
            "total_samples": self.tracker.get_total_samples(),
            "variants": {},
        }
        
        for variant_name in self.variant_names:
            variant = variant_map[variant_name]
            variant_stats = stats.get(variant_name, {})
            
            results["variants"][variant_name] = {
                "model_id": variant.model_id,
                "traffic_percentage": variant.traffic_percentage,
                "sample_count": self.tracker.get_sample_size(variant_name),
                "metrics": variant_stats,
            }
        
        results["timestamp"] = datetime.now().isoformat()
        
        return results

    def get_winner(self, metric_name: str = None) -> Optional[str]:
        """Determine winning variant.
        
        Args:
            metric_name: Metric to use for comparison (default: first metric)
            
        Returns:
            Name of winning variant, or None if not completed
        """
        if self.results is None or not self.results.get("variants"):
            return None
        
        if metric_name is None:
            if self.config.metrics:
                metric_name = self.config.metrics[0]
            else:
                return None
        
        best_variant = None
        best_value = float('-inf')
        
        for variant_name, variant_data in self.results["variants"].items():
            metrics = variant_data.get("metrics", {})
            if metric_name in metrics:
                mean_value = metrics[metric_name].get("mean", 0)
                if mean_value > best_value:
                    best_value = mean_value
                    best_variant = variant_name
        
        return best_variant

    def save_results(self, filepath: str) -> None:
        """Save experiment results to file.
        
        Args:
            filepath: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Complete experiment first.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved results to {filepath}")

    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get experiment results.
        
        Returns:
            Results dictionary or None if not completed
        """
        return self.results
