"""Model quantization for inference optimization.

Provides dynamic and static quantization strategies to reduce model size
and inference latency by 2-4x while maintaining accuracy.

Classes:
    CalibrationDataset: Dataset wrapper for quantization calibration
    DynamicQuantizer: Post-training dynamic quantization
    StaticQuantizer: Post-training static quantization
    QAT: Quantization-aware training
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig,
    default_dynamic_qconfig,
    default_qconfig,
    prepare,
    convert,
    quantize_dynamic,
)
from torch.utils.data import DataLoader, Dataset


class CalibrationDataset(Dataset):
    """Dataset wrapper for quantization calibration.
    
    Attributes:
        samples: List of sample tensors
        labels: Optional list of labels
    """

    def __init__(
        self,
        samples: List[torch.Tensor],
        labels: Optional[List[int]] = None,
    ):
        """Initialize calibration dataset.
        
        Args:
            samples: List of input tensors for calibration
            labels: Optional list of labels
        """
        self.samples = samples
        self.labels = labels or [None] * len(samples)
        assert len(self.samples) == len(self.labels)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """Get sample and label.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (sample, label)
        """
        return self.samples[idx], self.labels[idx]


class DynamicQuantizer:
    """Post-training dynamic quantization.
    
    Quantizes weights only (activations computed in float32).
    Faster calibration, lower accuracy impact, suitable for deployment.
    
    Attributes:
        model: PyTorch model to quantize
        device: Device to run quantization on
        quantization_config: Quantization configuration
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        dtype: torch.dtype = torch.qint8,
    ):
        """Initialize dynamic quantizer.
        
        Args:
            model: PyTorch model to quantize
            device: Device to run on ('cpu' or 'cuda')
            dtype: Target quantization dtype (default: qint8)
        """
        self.model = model.to(device)
        self.device = device
        self.dtype = dtype
        self.quantized_model = None
        self.metrics = {}

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization to model.
        
        Returns:
            Quantized model
        """
        start_time = time.time()
        
        # Apply dynamic quantization to linear layers
        self.quantized_model = quantize_dynamic(
            self.model,
            qconfig_spec={torch.nn.Linear},
            dtype=self.dtype,
        )
        
        quantization_time = time.time() - start_time
        
        # Calculate metrics
        original_size = self._calculate_model_size(self.model)
        quantized_size = self._calculate_model_size(self.quantized_model)
        
        self.metrics = {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size,
            "quantization_time_s": quantization_time,
        }
        
        return self.quantized_model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    def get_metrics(self) -> Dict[str, float]:
        """Get quantization metrics.
        
        Returns:
            Dictionary of metrics (size, compression ratio, time)
        """
        return self.metrics.copy()

    def benchmark_inference(
        self,
        test_input: torch.Tensor,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark quantized model inference latency.
        
        Args:
            test_input: Sample input tensor
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary of latency metrics (mean, std, min, max in ms)
        """
        if self.quantized_model is None:
            raise ValueError("Model not quantized. Call quantize() first.")
        
        self.quantized_model.eval()
        test_input = test_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.quantized_model(test_input)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.quantized_model(test_input)
                latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }


class StaticQuantizer:
    """Post-training static quantization with calibration.
    
    Requires representative calibration data to determine activation ranges.
    Higher accuracy impact but better performance than dynamic quantization.
    
    Attributes:
        model: PyTorch model to quantize
        device: Device to run on
        calibration_loader: DataLoader for calibration
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: str = "cpu",
    ):
        """Initialize static quantizer.
        
        Args:
            model: PyTorch model to quantize
            calibration_loader: DataLoader with representative data
            device: Device to run on
        """
        self.model = model.to(device)
        self.calibration_loader = calibration_loader
        self.device = device
        self.quantized_model = None
        self.metrics = {}

    def quantize(self) -> nn.Module:
        """Apply static quantization with calibration.
        
        Returns:
            Quantized model
        """
        start_time = time.time()
        
        # Set quantization config
        self.model.qconfig = default_qconfig
        
        # Prepare model with fake quantization
        prepared_model = prepare(self.model, inplace=False)
        
        # Calibrate on representative data
        prepared_model.eval()
        with torch.no_grad():
            for inputs, _ in self.calibration_loader:
                inputs = inputs.to(self.device)
                _ = prepared_model(inputs)
        
        # Convert to quantized model
        self.quantized_model = convert(prepared_model, inplace=False)
        
        quantization_time = time.time() - start_time
        
        # Calculate metrics
        original_size = self._calculate_model_size(self.model)
        quantized_size = self._calculate_model_size(self.quantized_model)
        
        self.metrics = {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size,
            "quantization_time_s": quantization_time,
            "calibration_samples": len(self.calibration_loader.dataset),
        }
        
        return self.quantized_model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    def get_metrics(self) -> Dict[str, float]:
        """Get quantization metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()


class QAT(nn.Module):
    """Quantization-Aware Training wrapper.
    
    Simulates quantization during training for better accuracy preservation.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Initialize QAT wrapper.
        
        Args:
            model: PyTorch model to train with quantization awareness
            device: Device to run on
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.model.qconfig = default_qconfig
        prepare(self.model, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)

    def convert(self) -> nn.Module:
        """Convert to quantized model.
        
        Returns:
            Quantized model
        """
        return convert(self.model, inplace=False)
