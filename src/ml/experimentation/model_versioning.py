"""Model versioning and deployment strategies.

Manages model versions, registry, and deployment rollout strategies
(canary, blue-green, A/B testing).

Classes:
    ModelVersion: Represents a versioned model
    ModelRegistry: Stores and manages model versions
    RolloutStrategy: Base class for rollout strategies
    CanaryRollout: Gradual traffic shift strategy
    BlueGreenDeployment: Atomic deployment strategy
    ABTestRollout: A/B test deployment strategy
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ModelStatus(Enum):
    """Model deployment status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class HealthStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelVersion:
    """Versioned model metadata.
    
    Attributes:
        model_id: Unique model identifier
        version: Version number/string
        status: Deployment status
        metrics: Model performance metrics
        created_at: Creation timestamp
        updated_at: Last update timestamp
        description: Description of changes
        tags: Custom tags for organization
    """
    model_id: str
    version: str
    status: ModelStatus = ModelStatus.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return asdict(self)


class ModelRegistry:
    """Registry for managing model versions.
    
    Stores model versions, enables easy rollback and comparison.
    
    Attributes:
        registry_path: Path to registry storage
        models: Dictionary of models by model_id
    """

    def __init__(self, registry_path: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            registry_path: Path to store registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, List[ModelVersion]] = {}
        self._load_registry()

    def register_model(self, model_version: ModelVersion) -> None:
        """Register a model version.
        
        Args:
            model_version: ModelVersion to register
        """
        if model_version.model_id not in self.models:
            self.models[model_version.model_id] = []
        
        # Check if version already exists
        existing = [v for v in self.models[model_version.model_id]
                   if v.version == model_version.version]
        if existing:
            raise ValueError(f"Version {model_version.version} already exists")
        
        self.models[model_version.model_id].append(model_version)
        self._save_registry()

    def get_model(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version.
        
        Args:
            model_id: Model ID
            version: Version string
            
        Returns:
            ModelVersion or None if not found
        """
        if model_id not in self.models:
            return None
        
        for model in self.models[model_id]:
            if model.version == version:
                return model
        
        return None

    def get_latest(self, model_id: str) -> Optional[ModelVersion]:
        """Get latest version of model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Latest ModelVersion or None
        """
        if model_id not in self.models or not self.models[model_id]:
            return None
        
        return self.models[model_id][-1]

    def get_production_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get current production model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Production ModelVersion or None
        """
        if model_id not in self.models:
            return None
        
        for model in reversed(self.models[model_id]):
            if model.status == ModelStatus.PRODUCTION:
                return model
        
        return None

    def list_models(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of ModelVersions
        """
        return self.models.get(model_id, [])

    def update_status(
        self,
        model_id: str,
        version: str,
        new_status: ModelStatus,
    ) -> None:
        """Update model status.
        
        Args:
            model_id: Model ID
            version: Version string
            new_status: New status
        """
        model = self.get_model(model_id, version)
        if model is None:
            raise ValueError(f"Model {model_id}:{version} not found")
        
        model.status = new_status
        model.updated_at = datetime.now().isoformat()
        self._save_registry()

    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str,
    ) -> Dict[str, any]:
        """Compare two model versions.
        
        Args:
            model_id: Model ID
            version1: First version
            version2: Second version
            
        Returns:
            Dictionary of differences
        """
        model1 = self.get_model(model_id, version1)
        model2 = self.get_model(model_id, version2)
        
        if model1 is None or model2 is None:
            raise ValueError("One or both models not found")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_v1": model1.metrics,
            "metrics_v2": model2.metrics,
            "metric_changes": {},
        }
        
        # Compute metric differences
        for metric in set(model1.metrics.keys()) | set(model2.metrics.keys()):
            v1 = model1.metrics.get(metric, 0)
            v2 = model2.metrics.get(metric, 0)
            comparison["metric_changes"][metric] = v2 - v1
        
        return comparison

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        data = {
            model_id: [m.to_dict() for m in versions]
            for model_id, versions in self.models.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if not registry_file.exists():
            return
        
        with open(registry_file, 'r') as f:
            data = json.load(f)
        
        for model_id, versions in data.items():
            self.models[model_id] = [
                ModelVersion(
                    model_id=v["model_id"],
                    version=v["version"],
                    status=ModelStatus(v["status"]),
                    metrics=v["metrics"],
                    created_at=v["created_at"],
                    updated_at=v["updated_at"],
                    description=v["description"],
                    tags=v["tags"],
                    parent_version=v.get("parent_version"),
                )
                for v in versions
            ]


class RolloutStrategy(ABC):
    """Base class for deployment rollout strategies."""

    @abstractmethod
    def get_variant_traffic(self, timestamp: float) -> Dict[str, float]:
        """Get traffic allocation for each variant at timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping variant names to traffic percentages
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if rollout is complete.
        
        Returns:
            True if rollout finished
        """
        pass


class CanaryRollout(RolloutStrategy):
    """Canary deployment: gradual traffic shift.
    
    Starts with small traffic % to new model, gradually increases
    based on metrics and time.
    
    Attributes:
        control_variant: Control variant name
        treatment_variant: Treatment variant name
        initial_traffic: Initial treatment traffic %
        target_traffic: Target treatment traffic %
        duration_seconds: Total rollout duration
        start_time: Rollout start timestamp
    """

    def __init__(
        self,
        control_variant: str,
        treatment_variant: str,
        initial_traffic: float = 0.05,
        target_traffic: float = 1.0,
        duration_seconds: float = 86400,
    ):
        """Initialize canary rollout.
        
        Args:
            control_variant: Name of stable variant
            treatment_variant: Name of new variant
            initial_traffic: Initial traffic % for treatment
            target_traffic: Target traffic % for treatment
            duration_seconds: Total rollout duration
        """
        self.control_variant = control_variant
        self.treatment_variant = treatment_variant
        self.initial_traffic = initial_traffic
        self.target_traffic = target_traffic
        self.duration_seconds = duration_seconds
        self.start_time = time.time()
        self.complete_time: Optional[float] = None

    def get_variant_traffic(self, timestamp: float) -> Dict[str, float]:
        """Get traffic allocation at timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary of variant traffic percentages
        """
        elapsed = timestamp - self.start_time
        
        if elapsed >= self.duration_seconds:
            self.complete_time = timestamp
            return {
                self.control_variant: 1.0 - self.target_traffic,
                self.treatment_variant: self.target_traffic,
            }
        
        # Linear increase from initial to target
        progress = elapsed / self.duration_seconds
        current_traffic = self.initial_traffic + (self.target_traffic - self.initial_traffic) * progress
        
        return {
            self.control_variant: 1.0 - current_traffic,
            self.treatment_variant: current_traffic,
        }

    def is_complete(self) -> bool:
        """Check if rollout complete.
        
        Returns:
            True if rollout finished
        """
        return self.complete_time is not None


class BlueGreenDeployment(RolloutStrategy):
    """Blue-green deployment: atomic switch.
    
    Deploys new version (green) alongside old (blue),
    then switches traffic atomically.
    
    Attributes:
        blue_variant: Current production variant
        green_variant: New variant to deploy
        switch_time: When to switch traffic
        switched: Whether switch has occurred
    """

    def __init__(
        self,
        blue_variant: str,
        green_variant: str,
        switch_time: float,
    ):
        """Initialize blue-green deployment.
        
        Args:
            blue_variant: Current production variant
            green_variant: New variant to deploy
            switch_time: Unix timestamp when to switch
        """
        self.blue_variant = blue_variant
        self.green_variant = green_variant
        self.switch_time = switch_time
        self.switched = False

    def get_variant_traffic(self, timestamp: float) -> Dict[str, float]:
        """Get traffic allocation at timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary of variant traffic percentages
        """
        if timestamp >= self.switch_time:
            self.switched = True
            return {
                self.blue_variant: 0.0,
                self.green_variant: 1.0,
            }
        else:
            return {
                self.blue_variant: 1.0,
                self.green_variant: 0.0,
            }

    def is_complete(self) -> bool:
        """Check if deployment complete.
        
        Returns:
            True if switched
        """
        return self.switched


class ABTestRollout(RolloutStrategy):
    """A/B test deployment: equal or custom traffic split.
    
    Maintains fixed traffic split between control and treatment
    for the duration of experiment.
    
    Attributes:
        control_variant: Control variant name
        treatment_variant: Treatment variant name
        control_traffic: Control traffic percentage
        end_time: When A/B test ends
    """

    def __init__(
        self,
        control_variant: str,
        treatment_variant: str,
        control_traffic: float = 0.5,
        duration_seconds: float = 604800,
    ):
        """Initialize A/B test rollout.
        
        Args:
            control_variant: Control variant name
            treatment_variant: Treatment variant name
            control_traffic: Traffic % for control (0-1)
            duration_seconds: Duration of A/B test
        """
        self.control_variant = control_variant
        self.treatment_variant = treatment_variant
        self.control_traffic = control_traffic
        self.treatment_traffic = 1.0 - control_traffic
        self.start_time = time.time()
        self.duration_seconds = duration_seconds
        self.end_time = self.start_time + duration_seconds
        self.finished = False

    def get_variant_traffic(self, timestamp: float) -> Dict[str, float]:
        """Get traffic allocation.
        
        Returns:
            Dictionary of fixed traffic percentages
        """
        if timestamp >= self.end_time:
            self.finished = True
        
        return {
            self.control_variant: self.control_traffic,
            self.treatment_variant: self.treatment_traffic,
        }

    def is_complete(self) -> bool:
        """Check if test complete.
        
        Returns:
            True if duration exceeded
        """
        return self.finished
