"""ML experimentation module.

Provides comprehensive A/B testing and experimentation framework:
- Experiment management and tracking
- Statistical significance testing
- Model versioning and rollout strategies
- Multi-variant experiments
"""

from src.ml.experimentation.experiment_manager import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentStatus,
    ExperimentTracker,
    ModelVariant,
)
from src.ml.experimentation.model_versioning import (
    ABTestRollout,
    BlueGreenDeployment,
    CanaryRollout,
    HealthStatus,
    ModelRegistry,
    ModelStatus,
    ModelVersion,
    RolloutStrategy,
)
from src.ml.experimentation.statistical_tests import (
    calculate_sample_size,
    calculate_statistical_power,
    chi_square,
    confidence_interval,
    is_statistically_significant,
    relative_lift,
    t_test,
)

__all__ = [
    # Experiment management
    "ExperimentManager",
    "ExperimentConfig",
    "ExperimentTracker",
    "ExperimentStatus",
    "ModelVariant",
    # Model versioning
    "ModelRegistry",
    "ModelVersion",
    "ModelStatus",
    "HealthStatus",
    # Rollout strategies
    "RolloutStrategy",
    "CanaryRollout",
    "BlueGreenDeployment",
    "ABTestRollout",
    # Statistical testing
    "t_test",
    "chi_square",
    "calculate_sample_size",
    "calculate_statistical_power",
    "confidence_interval",
    "relative_lift",
    "is_statistically_significant",
]
