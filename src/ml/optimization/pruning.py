"""Model pruning and compression strategies.

Provides structured and unstructured pruning techniques to reduce
model parameters and improve inference speed.

Classes:
    StructuredPruner: Channel/filter-level pruning
    UnstructuredPruner: Weight-level pruning
    PruningScheduler: Progressive pruning during training
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class StructuredPruner:
    """Channel/filter-level structured pruning.
    
    Removes entire channels or filters based on importance scores.
    Maintains model architecture and hardware efficiency.
    
    Attributes:
        model: PyTorch model to prune
        pruning_ratio: Target sparsity ratio (0-1)
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
    ):
        """Initialize structured pruner.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Target sparsity ratio (0.3 = 30% pruning)
        """
        if not 0 <= pruning_ratio <= 1:
            raise ValueError("pruning_ratio must be between 0 and 1")
        
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.metrics = {}

    def prune_by_importance(
        self,
        importance_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """Prune channels by importance score.
        
        Args:
            importance_fn: Function to compute importance (default: L1 norm)
            
        Returns:
            Pruned model
        """
        if importance_fn is None:
            importance_fn = self._l1_importance
        
        start_time = time.time()
        original_params = self._count_parameters(self.model)
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Compute importance for each output channel
                importance = importance_fn(module)
                
                # Determine threshold for pruning
                threshold = np.percentile(
                    importance,
                    self.pruning_ratio * 100,
                )
                
                # Create mask
                mask = importance > threshold
                
                # Apply structured pruning
                if isinstance(module, nn.Conv2d):
                    # Prune output channels in Conv2d
                    module.weight.data[~mask] = 0
                else:
                    # Prune output units in Linear
                    module.weight.data[~mask] = 0
        
        pruning_time = time.time() - start_time
        final_params = self._count_parameters(self.model)
        
        self.metrics = {
            "original_params": original_params,
            "pruned_params": final_params,
            "pruning_ratio_achieved": (original_params - final_params) / original_params,
            "pruning_time_s": pruning_time,
        }
        
        return self.model

    def _l1_importance(self, module: nn.Module) -> np.ndarray:
        """Compute L1 norm-based importance.
        
        Args:
            module: Network module
            
        Returns:
            Importance scores per channel
        """
        if isinstance(module, nn.Conv2d):
            # L1 norm per output channel
            return np.sum(np.abs(module.weight.data.cpu().numpy()), axis=(1, 2, 3))
        else:
            # L1 norm per output unit
            return np.sum(np.abs(module.weight.data.cpu().numpy()), axis=1)

    def _taylor_importance(self, module: nn.Module) -> np.ndarray:
        """Compute Taylor expansion-based importance.
        
        Args:
            module: Network module
            
        Returns:
            Importance scores
        """
        if not hasattr(module.weight, 'grad') or module.weight.grad is None:
            return self._l1_importance(module)
        
        weight = module.weight.data.cpu().numpy()
        grad = module.weight.grad.data.cpu().numpy()
        
        if isinstance(module, nn.Conv2d):
            taylor = np.sum(np.abs(weight * grad), axis=(1, 2, 3))
        else:
            taylor = np.sum(np.abs(weight * grad), axis=1)
        
        return taylor

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in model.parameters())

    def get_metrics(self) -> Dict[str, float]:
        """Get pruning metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()


class UnstructuredPruner:
    """Weight-level unstructured pruning.
    
    Removes individual weights based on magnitude or gradient-based importance.
    Maximum flexibility but may require specialized hardware for acceleration.
    
    Attributes:
        model: PyTorch model to prune
        pruning_ratio: Target sparsity ratio
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.5,
    ):
        """Initialize unstructured pruner.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Target sparsity ratio (0.5 = 50% pruning)
        """
        if not 0 <= pruning_ratio <= 1:
            raise ValueError("pruning_ratio must be between 0 and 1")
        
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.metrics = {}

    def prune_by_magnitude(self) -> nn.Module:
        """Prune weights by magnitude.
        
        Removes smallest weights to achieve target sparsity.
        
        Returns:
            Pruned model
        """
        start_time = time.time()
        original_params = self._count_parameters(self.model)
        
        # Collect all weights
        weights = []
        weight_refs = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                weights.append(param.data.abs().flatten())
                weight_refs.append(param)
        
        if not weights:
            raise ValueError("No weights found for pruning")
        
        # Determine global threshold
        all_weights = torch.cat(weights)
        threshold = torch.quantile(
            all_weights,
            torch.tensor(self.pruning_ratio, device=all_weights.device),
        )
        
        # Apply pruning
        for param in weight_refs:
            param.data *= (param.data.abs() > threshold).float()
        
        pruning_time = time.time() - start_time
        final_params = self._count_parameters(self.model)
        
        self.metrics = {
            "original_params": original_params,
            "pruned_params": final_params,
            "pruning_ratio_achieved": (original_params - final_params) / original_params,
            "pruning_time_s": pruning_time,
        }
        
        return self.model

    def prune_iterative(
        self,
        pruning_amount: float = 0.1,
        num_iterations: int = 5,
    ) -> nn.Module:
        """Iterative magnitude pruning.
        
        Gradually prunes model over multiple iterations for better accuracy.
        
        Args:
            pruning_amount: Sparsity per iteration
            num_iterations: Number of pruning iterations
            
        Returns:
            Pruned model
        """
        start_time = time.time()
        original_params = self._count_parameters(self.model)
        
        for iteration in range(num_iterations):
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'weight'):
                        prune.l1_unstructured(
                            module,
                            name='weight',
                            amount=pruning_amount,
                        )
        
        # Remove pruning reparameterization
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        pruning_time = time.time() - start_time
        final_params = self._count_parameters(self.model)
        
        self.metrics = {
            "original_params": original_params,
            "pruned_params": final_params,
            "pruning_ratio_achieved": (original_params - final_params) / original_params,
            "iterations": num_iterations,
            "pruning_time_s": pruning_time,
        }
        
        return self.model

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in model.parameters())

    def get_metrics(self) -> Dict[str, float]:
        """Get pruning metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()


class PruningScheduler:
    """Progressive pruning during training.
    
    Gradually increases sparsity during training for better accuracy preservation.
    
    Attributes:
        model: PyTorch model to prune
        start_sparsity: Initial sparsity ratio
        end_sparsity: Target sparsity ratio
        total_steps: Total training steps
    """

    def __init__(
        self,
        model: nn.Module,
        start_sparsity: float = 0.0,
        end_sparsity: float = 0.9,
        total_steps: int = 1000,
    ):
        """Initialize pruning scheduler.
        
        Args:
            model: PyTorch model
            start_sparsity: Initial sparsity
            end_sparsity: Target sparsity
            total_steps: Total training steps
        """
        self.model = model
        self.start_sparsity = start_sparsity
        self.end_sparsity = end_sparsity
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> float:
        """Update pruning ratio for current step.
        
        Returns:
            Current sparsity ratio
        """
        progress = self.current_step / self.total_steps
        current_sparsity = (
            self.start_sparsity +
            (self.end_sparsity - self.start_sparsity) * progress
        )
        
        self._apply_sparsity(current_sparsity)
        self.current_step += 1
        
        return current_sparsity

    def _apply_sparsity(self, sparsity: float) -> None:
        """Apply sparsity to model weights.
        
        Args:
            sparsity: Target sparsity ratio
        """
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    threshold = torch.quantile(
                        weight.abs(),
                        torch.tensor(sparsity, device=weight.device),
                    )
                    module.weight.data *= (weight.abs() > threshold).float()
