"""Inference pipeline optimization with batching and caching.

Provides efficient inference serving with batch processing,
prediction caching, and latency optimization.

Classes:
    PredictionCache: LRU cache for model predictions
    BatchProcessor: Dynamic batching for inference
    InferencePipeline: End-to-end optimized inference
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class PredictionCache:
    """LRU prediction cache for repeated inputs.
    
    Caches model outputs to avoid redundant inference on identical inputs.
    Useful for deduplication and repeated queries.
    
    Attributes:
        max_size: Maximum number of cached predictions
        ttl: Time-to-live in seconds (0 = no expiration)
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: Optional[float] = None,
    ):
        """Initialize prediction cache.
        
        Args:
            max_size: Maximum number of entries in cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_input(self, input_tensor: torch.Tensor) -> str:
        """Create hash of input tensor.
        
        Args:
            input_tensor: Input tensor to hash
            
        Returns:
            Hexadecimal hash string
        """
        data = input_tensor.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached prediction.
        
        Args:
            key: Cache key (hash of input)
            
        Returns:
            Cached prediction or None if not found/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        prediction, timestamp = self.cache[key]
        
        # Check TTL
        if self.ttl is not None:
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return prediction

    def put(self, key: str, value: Any) -> None:
        """Store prediction in cache.
        
        Args:
            key: Cache key
            value: Prediction to cache
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = (value, time.time())
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get_hit_rate(self) -> float:
        """Get cache hit rate.
        
        Returns:
            Hit rate between 0 and 1
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def clear(self) -> None:
        """Clear all cached predictions."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics.
        
        Returns:
            Dictionary of stats (hits, misses, hit rate, size)
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }


class BatchProcessor:
    """Dynamic batching for efficient inference.
    
    Accumulates requests and processes them in batches
    to maximize GPU utilization and throughput.
    
    Attributes:
        max_batch_size: Maximum batch size
        max_wait_time: Maximum wait time before processing (seconds)
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
    ):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_time: Max wait time before processing incomplete batch
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue: List[torch.Tensor] = []
        self.queue_time = time.time()

    def add_request(self, input_tensor: torch.Tensor) -> bool:
        """Add request to batch queue.
        
        Args:
            input_tensor: Input tensor to add
            
        Returns:
            True if batch is ready, False otherwise
        """
        self.queue.append(input_tensor)
        
        # Check if batch is ready
        if len(self.queue) >= self.max_batch_size:
            return True
        
        # Check if we've waited long enough
        if time.time() - self.queue_time > self.max_wait_time:
            return True
        
        return False

    def get_batch(self) -> Optional[torch.Tensor]:
        """Get current batch.
        
        Returns:
            Stacked batch tensor or None if queue empty
        """
        if not self.queue:
            return None
        
        batch = torch.stack(self.queue)
        self.queue = []
        self.queue_time = time.time()
        
        return batch

    def queue_size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of requests in queue
        """
        return len(self.queue)


class InferencePipeline:
    """End-to-end optimized inference pipeline.
    
    Combines model, caching, batching, and preprocessing for
    efficient inference with minimal latency.
    
    Attributes:
        model: PyTorch model
        cache: Prediction cache
        batch_processor: Batch processor
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        cache_size: int = 10000,
        batch_size: int = 32,
        cache_ttl: Optional[float] = None,
    ):
        """Initialize inference pipeline.
        
        Args:
            model: PyTorch model
            device: Device to run on
            cache_size: Prediction cache size
            batch_size: Batch processor size
            cache_ttl: Cache time-to-live in seconds
        """
        self.model = model.to(device).eval()
        self.device = device
        self.cache = PredictionCache(max_size=cache_size, ttl=cache_ttl)
        self.batch_processor = BatchProcessor(max_batch_size=batch_size)
        
        self.latencies = []
        self.throughput_samples = []

    def infer(
        self,
        input_tensor: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run inference with caching and optimization.
        
        Args:
            input_tensor: Input tensor
            use_cache: Whether to use prediction cache
            
        Returns:
            Tuple of (prediction, metadata dict with timing info)
        """
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cache_key = self.cache._hash_input(input_tensor)
            cached_pred = self.cache.get(cache_key)
            if cached_pred is not None:
                return cached_pred, {"cached": True, "latency_ms": 0.0}
        else:
            cache_key = None
        
        # Run inference
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Cache result
        if use_cache and cache_key:
            self.cache.put(cache_key, prediction)
        
        latency_ms = (time.time() - start_time) * 1000
        self.latencies.append(latency_ms)
        
        return prediction, {
            "cached": False,
            "latency_ms": latency_ms,
        }

    def infer_batch(
        self,
        batch: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run inference on batch.
        
        Args:
            batch: Batch of inputs (batch_size, ...)
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (predictions, metadata)
        """
        start_time = time.time()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        latency_ms = (time.time() - start_time) * 1000
        throughput = batch.size(0) / (latency_ms / 1000)
        
        self.latencies.append(latency_ms / batch.size(0))
        self.throughput_samples.append(throughput)
        
        return predictions, {
            "batch_size": batch.size(0),
            "latency_ms": latency_ms,
            "throughput": throughput,
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        
        stats = {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
        }
        
        if self.throughput_samples:
            throughput = np.array(self.throughput_samples)
            stats.update({
                "mean_throughput": float(np.mean(throughput)),
                "max_throughput": float(np.max(throughput)),
            })
        
        stats.update(self.cache.get_stats())
        
        return stats

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.latencies = []
        self.throughput_samples = []
        self.cache.clear()
