"""Real-time model performance monitoring.

Tracks model accuracy, latency, throughput, and other metrics
to detect performance degradation and trigger alerts.

Classes:
    PerformanceAlert: Alert triggered when metrics exceed thresholds
    ModelPerformanceMonitor: Tracks model metrics over time
    MetricTracker: Individual metric tracking with aggregation
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert raised when metrics exceed thresholds.
    
    Attributes:
        metric_name: Name of metric that triggered alert
        current_value: Current metric value
        threshold: Threshold that was exceeded
        severity: Alert severity level
        timestamp: When alert was raised
        description: Alert description
    """
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    timestamp: datetime
    description: str = ""

    def __str__(self) -> str:
        """String representation."""
        return (f"[{self.severity.value.upper()}] {self.metric_name}: "
                f"{self.current_value:.4f} > {self.threshold:.4f}")


class MetricTracker:
    """Tracks individual metrics with aggregation.
    
    Attributes:
        name: Metric name
        max_samples: Maximum number of samples to keep
        aggregation_fn: Function to aggregate samples
    """

    def __init__(
        self,
        name: str,
        max_samples: int = 1000,
        aggregation_fn: Callable = np.mean,
    ):
        """Initialize metric tracker.
        
        Args:
            name: Metric name
            max_samples: Max samples to store
            aggregation_fn: Function to aggregate (mean, median, max, etc.)
        """
        self.name = name
        self.max_samples = max_samples
        self.aggregation_fn = aggregation_fn
        self.values: Deque[float] = deque(maxlen=max_samples)
        self.timestamps: Deque[float] = deque(maxlen=max_samples)

    def record(self, value: float, timestamp: Optional[float] = None) -> None:
        """Record metric value.
        
        Args:
            value: Metric value
            timestamp: Timestamp (default: current time)
        """
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

    def get_aggregate(self) -> float:
        """Get aggregated metric value.
        
        Returns:
            Aggregated value
        """
        if not self.values:
            return 0.0
        return float(self.aggregation_fn(self.values))

    def get_stats(self) -> Dict[str, float]:
        """Get metric statistics.
        
        Returns:
            Dictionary of stats (mean, std, min, max, median, p95)
        """
        if not self.values:
            return {}
        
        values = np.array(self.values)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "count": len(self.values),
        }

    def get_trend(self, window_size: int = 10) -> str:
        """Get trend direction (improving/degrading/stable).
        
        Args:
            window_size: Size of window for trend calculation
            
        Returns:
            Trend string ("improving", "degrading", or "stable")
        """
        if len(self.values) < window_size:
            return "insufficient_data"
        
        recent = list(self.values)[-window_size:]
        early = list(self.values)[-window_size*2:-window_size]
        
        recent_mean = np.mean(recent)
        early_mean = np.mean(early)
        
        if recent_mean < early_mean:
            return "improving"
        elif recent_mean > early_mean:
            return "degrading"
        else:
            return "stable"

    def clear(self) -> None:
        """Clear all recorded values."""
        self.values.clear()
        self.timestamps.clear()


class ModelPerformanceMonitor:
    """Monitor model performance metrics in real-time.
    
    Tracks accuracy, latency, throughput and other metrics,
    detects performance degradation and raises alerts.
    
    Attributes:
        model_name: Name of model being monitored
        metrics: Dictionary of MetricTracker instances
        alerts: List of raised alerts
    """

    def __init__(
        self,
        model_name: str,
        max_samples: int = 1000,
    ):
        """Initialize performance monitor.
        
        Args:
            model_name: Name of model to monitor
            max_samples: Max samples to keep per metric
        """
        self.model_name = model_name
        self.max_samples = max_samples
        self.metrics: Dict[str, MetricTracker] = {}
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}

    def register_metric(
        self,
        name: str,
        aggregation_fn: Callable = np.mean,
    ) -> None:
        """Register a metric to track.
        
        Args:
            name: Metric name
            aggregation_fn: Function to aggregate values
        """
        self.metrics[name] = MetricTracker(name, self.max_samples, aggregation_fn)

    def set_threshold(
        self,
        metric_name: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> None:
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of metric
            threshold: Threshold value
            severity: Alert severity level
        """
        if metric_name not in self.metrics:
            self.register_metric(metric_name)
        
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name]["value"] = threshold
        self.thresholds[metric_name]["severity"] = severity

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> Optional[PerformanceAlert]:
        """Record metric value and check thresholds.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            timestamp: Timestamp (default: current time)
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if metric_name not in self.metrics:
            self.register_metric(metric_name)
        
        # Record value
        self.metrics[metric_name].record(value, timestamp)
        
        # Check threshold
        alert = self._check_threshold(metric_name, value)
        if alert:
            self.alerts.append(alert)
        
        return alert

    def record_accuracy(
        self,
        accuracy: float,
        threshold: float = 0.85,
    ) -> Optional[PerformanceAlert]:
        """Record model accuracy.
        
        Args:
            accuracy: Accuracy value (0-1)
            threshold: Alert threshold
            
        Returns:
            Alert if accuracy below threshold
        """
        return self.record_metric("accuracy", accuracy)

    def record_latency(
        self,
        latency_ms: float,
        threshold: float = 100.0,
    ) -> Optional[PerformanceAlert]:
        """Record inference latency.
        
        Args:
            latency_ms: Latency in milliseconds
            threshold: Alert threshold
            
        Returns:
            Alert if latency exceeds threshold
        """
        return self.record_metric("latency_ms", latency_ms)

    def record_throughput(
        self,
        throughput: float,
        threshold: float = 100.0,
    ) -> Optional[PerformanceAlert]:
        """Record inference throughput.
        
        Args:
            throughput: Throughput (samples/sec)
            threshold: Alert threshold (minimum)
            
        Returns:
            Alert if throughput below threshold
        """
        return self.record_metric("throughput", throughput)

    def _check_threshold(
        self,
        metric_name: str,
        value: float,
    ) -> Optional[PerformanceAlert]:
        """Check if value exceeds threshold.
        
        Args:
            metric_name: Name of metric
            value: Current value
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if metric_name not in self.thresholds:
            return None
        
        threshold_info = self.thresholds[metric_name]
        threshold = threshold_info["value"]
        severity = threshold_info.get("severity", AlertSeverity.WARNING)
        
        # For accuracy and throughput, lower is bad
        # For latency, higher is bad
        if metric_name in ["accuracy"]:
            triggered = value < threshold
        elif metric_name in ["latency_ms", "error_rate"]:
            triggered = value > threshold
        else:
            triggered = value > threshold
        
        if triggered:
            return PerformanceAlert(
                metric_name=metric_name,
                current_value=value,
                threshold=threshold,
                severity=severity,
                timestamp=datetime.now(),
                description=f"{metric_name} exceeded threshold",
            )
        
        return None

    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dictionary of statistics
        """
        if metric_name not in self.metrics:
            return {}
        
        tracker = self.metrics[metric_name]
        stats = tracker.get_stats()
        stats["trend"] = tracker.get_trend()
        
        if metric_name in self.thresholds:
            stats["threshold"] = self.thresholds[metric_name]["value"]
        
        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics.
        
        Returns:
            Dictionary mapping metric names to stats
        """
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }

    def get_recent_alerts(
        self,
        limit: int = 10,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> List[PerformanceAlert]:
        """Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            min_severity: Minimum severity level to include
            
        Returns:
            List of recent alerts
        """
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
        }
        
        filtered = [
            a for a in self.alerts
            if severity_order[a.severity] >= severity_order[min_severity]
        ]
        
        return filtered[-limit:]

    def clear_alerts(self) -> None:
        """Clear all recorded alerts."""
        self.alerts.clear()

    def reset_metrics(self) -> None:
        """Clear all metric data."""
        for tracker in self.metrics.values():
            tracker.clear()
        self.alerts.clear()

    def get_health_report(self) -> Dict[str, Any]:
        """Get overall model health report.
        
        Returns:
            Dictionary with health status and metrics
        """
        all_stats = self.get_all_stats()
        recent_alerts = self.get_recent_alerts(limit=5)
        
        # Determine overall health
        if recent_alerts:
            health = "unhealthy" if any(
                a.severity == AlertSeverity.CRITICAL
                for a in recent_alerts
            ) else "degraded"
        else:
            health = "healthy"
        
        return {
            "model_name": self.model_name,
            "health": health,
            "metrics": all_stats,
            "recent_alerts": [
                {
                    "metric": a.metric_name,
                    "value": a.current_value,
                    "threshold": a.threshold,
                    "severity": a.severity.value,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in recent_alerts
            ],
            "timestamp": datetime.now().isoformat(),
        }
