"""Cost analysis and optimization tools.

Tracks resource usage and ML infrastructure costs,
provides recommendations for cost optimization.

Classes:
    ResourceUsage: Resource usage metrics
    ResourceTracker: Tracks CPU, GPU, memory usage
    CostEstimator: Estimates operational costs
    OptimizationAdvisor: Recommends cost optimizations
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class CloudProvider(Enum):
    """Cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREMISE = "on_premise"


@dataclass
class ResourceUsage:
    """Resource usage metrics.
    
    Attributes:
        timestamp: When usage was recorded
        cpu_percent: CPU usage percentage (0-100)
        gpu_percent: GPU usage percentage (0-100)
        memory_mb: Memory usage in MB
        storage_gb: Storage usage in GB
        network_mb: Network transfer in MB
    """
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_mb: float = 0.0
    storage_gb: float = 0.0
    network_mb: float = 0.0


class ResourceTracker:
    """Tracks resource usage over time.
    
    Records CPU, GPU, memory, storage, and network usage
    to enable cost analysis.
    
    Attributes:
        max_samples: Maximum samples to keep
        usage_history: Historical usage data
    """

    def __init__(self, max_samples: int = 10000):
        """Initialize resource tracker.
        
        Args:
            max_samples: Maximum samples to store
        """
        self.max_samples = max_samples
        self.usage_history: List[ResourceUsage] = []
        self.start_time = time.time()

    def record(
        self,
        cpu_percent: float = 0.0,
        gpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        storage_gb: float = 0.0,
        network_mb: float = 0.0,
    ) -> None:
        """Record resource usage.
        
        Args:
            cpu_percent: CPU usage (0-100)
            gpu_percent: GPU usage (0-100)
            memory_mb: Memory in MB
            storage_gb: Storage in GB
            network_mb: Network transfer in MB
        """
        usage = ResourceUsage(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            memory_mb=memory_mb,
            storage_gb=storage_gb,
            network_mb=network_mb,
        )
        
        self.usage_history.append(usage)
        
        # Prune old data
        if len(self.usage_history) > self.max_samples:
            self.usage_history = self.usage_history[-self.max_samples:]

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get resource usage statistics.
        
        Returns:
            Dictionary of stats for each resource type
        """
        if not self.usage_history:
            return {}
        
        cpu_values = np.array([u.cpu_percent for u in self.usage_history])
        gpu_values = np.array([u.gpu_percent for u in self.usage_history])
        mem_values = np.array([u.memory_mb for u in self.usage_history])
        storage_values = np.array([u.storage_gb for u in self.usage_history])
        network_values = np.array([u.network_mb for u in self.usage_history])
        
        def compute_stats(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)),
            }
        
        return {
            "cpu": compute_stats(cpu_values),
            "gpu": compute_stats(gpu_values),
            "memory_mb": compute_stats(mem_values),
            "storage_gb": compute_stats(storage_values),
            "network_mb": compute_stats(network_values),
        }

    def get_peak_usage(self) -> ResourceUsage:
        """Get peak usage across all metrics.
        
        Returns:
            ResourceUsage with maximum values
        """
        if not self.usage_history:
            return ResourceUsage()
        
        max_cpu = max(u.cpu_percent for u in self.usage_history)
        max_gpu = max(u.gpu_percent for u in self.usage_history)
        max_mem = max(u.memory_mb for u in self.usage_history)
        max_storage = max(u.storage_gb for u in self.usage_history)
        max_network = max(u.network_mb for u in self.usage_history)
        
        return ResourceUsage(
            cpu_percent=max_cpu,
            gpu_percent=max_gpu,
            memory_mb=max_mem,
            storage_gb=max_storage,
            network_mb=max_network,
        )

    def get_duration_hours(self) -> float:
        """Get tracking duration in hours.
        
        Returns:
            Duration in hours
        """
        return (time.time() - self.start_time) / 3600

    def clear(self) -> None:
        """Clear usage history."""
        self.usage_history.clear()
        self.start_time = time.time()


class CostEstimator:
    """Estimates operational costs based on resource usage.
    
    Attributes:
        provider: Cloud provider
        pricing: Pricing configuration per resource type
    """

    # Hourly pricing for major cloud providers (in USD)
    DEFAULT_PRICING = {
        CloudProvider.AWS: {
            "cpu_per_core_hour": 0.04,
            "gpu_per_hour": 0.30,  # GPU hours
            "memory_per_gb_hour": 0.01,
            "storage_per_gb_month": 0.023,
            "network_per_gb": 0.02,
        },
        CloudProvider.GCP: {
            "cpu_per_core_hour": 0.033,
            "gpu_per_hour": 0.35,
            "memory_per_gb_hour": 0.0044,
            "storage_per_gb_month": 0.020,
            "network_per_gb": 0.012,
        },
        CloudProvider.AZURE: {
            "cpu_per_core_hour": 0.033,
            "gpu_per_hour": 0.35,
            "memory_per_gb_hour": 0.004,
            "storage_per_gb_month": 0.0184,
            "network_per_gb": 0.017,
        },
        CloudProvider.ON_PREMISE: {
            "cpu_per_core_hour": 0.005,
            "gpu_per_hour": 0.08,
            "memory_per_gb_hour": 0.0008,
            "storage_per_gb_month": 0.001,
            "network_per_gb": 0.0,
        },
    }

    def __init__(
        self,
        provider: CloudProvider = CloudProvider.AWS,
        n_cpu_cores: int = 8,
        n_gpu: int = 1,
        total_memory_gb: int = 32,
        custom_pricing: Optional[Dict] = None,
    ):
        """Initialize cost estimator.
        
        Args:
            provider: Cloud provider
            n_cpu_cores: Number of CPU cores
            n_gpu: Number of GPUs
            total_memory_gb: Total memory in GB
            custom_pricing: Custom pricing override
        """
        self.provider = provider
        self.n_cpu_cores = n_cpu_cores
        self.n_gpu = n_gpu
        self.total_memory_gb = total_memory_gb
        
        self.pricing = self.DEFAULT_PRICING[provider].copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def estimate_hourly_cost(self, resource_usage: ResourceUsage) -> float:
        """Estimate hourly cost for given resource usage.
        
        Args:
            resource_usage: ResourceUsage metrics
            
        Returns:
            Estimated hourly cost in USD
        """
        cost = 0.0
        
        # CPU cost (percentage of cores used)
        cpu_utilization = resource_usage.cpu_percent / 100.0
        cpu_cost = (self.n_cpu_cores * cpu_utilization *
                   self.pricing["cpu_per_core_hour"])
        cost += cpu_cost
        
        # GPU cost (percentage of GPUs used)
        gpu_utilization = resource_usage.gpu_percent / 100.0
        gpu_cost = (self.n_gpu * gpu_utilization *
                   self.pricing["gpu_per_hour"])
        cost += gpu_cost
        
        # Memory cost (percentage of memory used)
        memory_utilization = resource_usage.memory_mb / (self.total_memory_gb * 1024)
        memory_cost = (self.total_memory_gb * memory_utilization *
                      self.pricing["memory_per_gb_hour"])
        cost += memory_cost
        
        # Network cost
        network_cost = resource_usage.network_mb / 1024 * self.pricing["network_per_gb"]
        cost += network_cost
        
        return cost

    def estimate_total_cost(
        self,
        usage_history: List[ResourceUsage],
    ) -> Dict[str, float]:
        """Estimate total cost from usage history.
        
        Args:
            usage_history: List of ResourceUsage records
            
        Returns:
            Dictionary of cost breakdown
        """
        if not usage_history:
            return {}
        
        # Estimate time per record (assume 1 minute intervals)
        time_per_record_hours = 1 / 60
        
        hourly_costs = [
            self.estimate_hourly_cost(usage) * time_per_record_hours
            for usage in usage_history
        ]
        
        total_cost = sum(hourly_costs)
        duration_hours = len(usage_history) * time_per_record_hours
        
        # Storage cost (monthly)
        if usage_history:
            avg_storage_gb = np.mean([u.storage_gb for u in usage_history])
            storage_cost = avg_storage_gb * self.pricing["storage_per_gb_month"]
        else:
            storage_cost = 0.0
        
        return {
            "total_compute_cost": total_cost,
            "storage_cost": storage_cost,
            "total_cost": total_cost + storage_cost,
            "duration_hours": duration_hours,
            "cost_per_hour": total_cost / max(duration_hours, 1),
        }

    def compare_providers(
        self,
        usage_history: List[ResourceUsage],
    ) -> Dict[str, Dict[str, float]]:
        """Compare costs across cloud providers.
        
        Args:
            usage_history: Resource usage history
            
        Returns:
            Dictionary of cost estimates per provider
        """
        comparison = {}
        
        for provider in CloudProvider:
            estimator = CostEstimator(provider, self.n_cpu_cores, self.n_gpu, self.total_memory_gb)
            comparison[provider.value] = estimator.estimate_total_cost(usage_history)
        
        return comparison


class OptimizationAdvisor:
    """Recommends cost optimization strategies.
    
    Analyzes resource usage patterns and suggests optimizations.
    
    Attributes:
        tracker: ResourceTracker instance
        estimator: CostEstimator instance
    """

    def __init__(
        self,
        tracker: ResourceTracker,
        estimator: CostEstimator,
    ):
        """Initialize optimization advisor.
        
        Args:
            tracker: ResourceTracker for usage data
            estimator: CostEstimator for cost calculations
        """
        self.tracker = tracker
        self.estimator = estimator

    def get_recommendations(self) -> List[Dict[str, any]]:
        """Get cost optimization recommendations.
        
        Returns:
            List of recommendations with priority and estimated savings
        """
        recommendations = []
        stats = self.tracker.get_statistics()
        
        if not stats:
            return recommendations
        
        # Check CPU utilization
        cpu_stats = stats.get("cpu", {})
        cpu_mean = cpu_stats.get("mean", 0)
        
        if cpu_mean < 20:
            recommendations.append({
                "category": "compute_sizing",
                "title": "Reduce CPU cores",
                "description": "CPU utilization is low. Consider reducing core count.",
                "priority": "high",
                "estimated_savings_percent": 20,
            })
        
        # Check GPU utilization
        gpu_stats = stats.get("gpu", {})
        gpu_mean = gpu_stats.get("mean", 0)
        
        if gpu_mean < 30:
            recommendations.append({
                "category": "compute_sizing",
                "title": "Optimize GPU allocation",
                "description": "GPU utilization is low. Consider fewer GPUs or spot instances.",
                "priority": "high",
                "estimated_savings_percent": 50,
            })
        
        # Check memory utilization
        mem_stats = stats.get("memory_mb", {})
        mem_mean = mem_stats.get("mean", 0)
        total_mem_mb = self.estimator.total_memory_gb * 1024
        mem_utilization = mem_mean / total_mem_mb if total_mem_mb > 0 else 0
        
        if mem_utilization < 0.3:
            recommendations.append({
                "category": "compute_sizing",
                "title": "Reduce memory allocation",
                "description": "Memory utilization is low. Consider reducing allocated memory.",
                "priority": "medium",
                "estimated_savings_percent": 20,
            })
        
        # Check for idle time
        if cpu_mean < 5 and gpu_mean < 5:
            recommendations.append({
                "category": "scheduling",
                "title": "Use spot/preemptible instances",
                "description": "System has periods of low utilization. Spot instances could save 50-90%.",
                "priority": "high",
                "estimated_savings_percent": 70,
            })
        
        # Check storage usage
        storage_stats = stats.get("storage_gb", {})
        storage_mean = storage_stats.get("mean", 0)
        
        if storage_mean > 500:
            recommendations.append({
                "category": "storage",
                "title": "Implement data cleanup",
                "description": f"High storage usage ({storage_mean:.0f}GB). Implement retention policies.",
                "priority": "medium",
                "estimated_savings_percent": 30,
            })
        
        # Reserved instances recommendation
        if self.estimator.provider != CloudProvider.ON_PREMISE:
            recommendations.append({
                "category": "commitment",
                "title": "Use reserved instances",
                "description": "Purchase reserved instances for predictable workloads. Can save 30-50%.",
                "priority": "medium",
                "estimated_savings_percent": 40,
            })
        
        return recommendations

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by resource type.
        
        Returns:
            Dictionary of costs by component
        """
        if not self.tracker.usage_history:
            return {}
        
        return self.estimator.estimate_total_cost(self.tracker.usage_history)

    def get_potential_savings(self) -> float:
        """Calculate potential savings with recommended optimizations.
        
        Returns:
            Estimated savings percentage
        """
        recommendations = self.get_recommendations()
        if not recommendations:
            return 0.0
        
        # Calculate total potential savings (conservative estimate)
        max_savings = 0.0
        for rec in recommendations:
            if rec.get("priority") == "high":
                max_savings += rec.get("estimated_savings_percent", 0) * 0.5
            else:
                max_savings += rec.get("estimated_savings_percent", 0) * 0.25
        
        return min(max_savings, 90.0)  # Cap at 90%
