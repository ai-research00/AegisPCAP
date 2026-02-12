"""Model optimization module.

Provides techniques for optimizing neural network models:
- Quantization: Reduce precision and model size
- Pruning: Remove less important weights
- Distillation: Transfer knowledge from large to small models
- Inference optimization: Batching, caching, preprocessing
"""

from src.ml.optimization.distillation import (
    DistillationLoss,
    KnowledgeDistiller,
    TemperatureScaling,
)
from src.ml.optimization.inference_pipeline import (
    BatchProcessor,
    InferencePipeline,
    PredictionCache,
)
from src.ml.optimization.pruning import (
    PruningScheduler,
    StructuredPruner,
    UnstructuredPruner,
)
from src.ml.optimization.quantization import (
    CalibrationDataset,
    DynamicQuantizer,
    QAT,
    StaticQuantizer,
)

__all__ = [
    # Quantization
    "CalibrationDataset",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QAT",
    # Pruning
    "StructuredPruner",
    "UnstructuredPruner",
    "PruningScheduler",
    # Distillation
    "KnowledgeDistiller",
    "DistillationLoss",
    "TemperatureScaling",
    # Inference
    "InferencePipeline",
    "PredictionCache",
    "BatchProcessor",
]
