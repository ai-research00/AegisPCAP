"""
Explainable AI (XAI) Module for AegisPCAP Phase 12

Provides SHAP, LIME, attention visualization, counterfactual explanations,
and feature interaction analysis for model interpretability.

Author: AegisPCAP Development
Date: February 5, 2026
Version: 1.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import warnings

# Import SHAP and LIME
try:
    import shap
except ImportError:
    shap = None
    
try:
    from lime import lime_tabular
except ImportError:
    lime_tabular = None

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Explanation:
    """Container for model explanations."""
    prediction: float
    confidence: float
    feature_importances: Dict[str, float]
    explanation_method: str
    additional_context: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert explanation to dictionary."""
        return {
            "prediction": float(self.prediction),
            "confidence": float(self.confidence),
            "feature_importances": self.feature_importances,
            "explanation_method": self.explanation_method,
            "additional_context": self.additional_context or {}
        }


@dataclass
class CounterfactualExample:
    """Counterfactual explanation for a prediction."""
    original_instance: np.ndarray
    counterfactual_instance: np.ndarray
    changes: Dict[str, float]  # Feature: change amount
    num_changes: int
    new_prediction: float
    plausibility_score: float


# ============================================================================
# SHAP Explanations
# ============================================================================

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for feature importance.
    
    Provides theoretically sound feature importance based on game theory.
    """
    
    def __init__(self, model, background_data: Optional[np.ndarray] = None,
                background_samples: int = 100):
        """
        Args:
            model: Trained model (sklearn or PyTorch)
            background_data: Background dataset for SHAP
            background_samples: Number of background samples to use
        """
        self.model = model
        self.background_samples = background_samples
        
        # Initialize SHAP explainer
        if shap is None:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.explainer = None
        else:
            if isinstance(model, torch.nn.Module):
                # For PyTorch models
                self.explainer = shap.DeepExplainer(
                    model,
                    torch.FloatTensor(background_data[:background_samples])
                    if background_data is not None else None
                )
            else:
                # For sklearn models (tree-based)
                try:
                    self.explainer = shap.TreeExplainer(model)
                except:
                    self.explainer = shap.KernelExplainer(
                        model.predict,
                        background_data[:background_samples]
                        if background_data is not None else None
                    )
        
        self.background_data = background_data
        
    def explain(self, instance: np.ndarray) -> Explanation:
        """
        Explain prediction for a single instance.
        
        Args:
            instance: Input instance to explain
            
        Returns:
            explanation: Explanation object with feature importances
        """
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return Explanation(
                prediction=0.0,
                confidence=0.0,
                feature_importances={},
                explanation_method="shap_unavailable"
            )
        
        try:
            if isinstance(self.model, torch.nn.Module):
                instance_t = torch.FloatTensor(instance).unsqueeze(0)
                shap_values = self.explainer.shap_values(instance_t)
                prediction = self.model(instance_t).detach().numpy()[0]
            else:
                shap_values = self.explainer.shap_values(instance.reshape(1, -1))
                prediction = self.model.predict(instance.reshape(1, -1))[0]
            
            # Handle multi-class (list of arrays)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Flatten if needed
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            # Create feature importance dict
            feature_importances = {
                f"feature_{i}": float(abs(shap_values[i]))
                for i in range(len(shap_values))
            }
            
            # Normalize importances to sum to 1
            total_importance = sum(feature_importances.values())
            if total_importance > 0:
                feature_importances = {
                    k: v / total_importance
                    for k, v in feature_importances.items()
                }
            
            return Explanation(
                prediction=float(prediction) if not isinstance(prediction, np.ndarray) else float(prediction[0]),
                confidence=float(np.abs(shap_values).sum()),
                feature_importances=feature_importances,
                explanation_method="shap",
                additional_context={"shap_values": shap_values.tolist()}
            )
        
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return Explanation(
                prediction=0.0,
                confidence=0.0,
                feature_importances={},
                explanation_method="shap_error"
            )


# ============================================================================
# LIME Explanations
# ============================================================================

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).
    
    Explains predictions by fitting local linear models.
    """
    
    def __init__(self, model, training_data: np.ndarray,
                feature_names: Optional[List[str]] = None,
                mode: str = "classification"):
        """
        Args:
            model: Prediction function
            training_data: Training data for learning distribution
            feature_names: Names of features
            mode: "classification" or "regression"
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
        self.mode = mode
        
        if lime_tabular is None:
            logger.warning("LIME not installed. Install with: pip install lime")
            self.explainer = None
        else:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                mode=mode,
                verbose=False
            )
    
    def explain(self, instance: np.ndarray, num_features: int = 10) -> Explanation:
        """
        Explain prediction using LIME.
        
        Args:
            instance: Instance to explain
            num_features: Number of top features to explain
            
        Returns:
            explanation: Explanation object
        """
        if self.explainer is None:
            logger.error("LIME explainer not initialized")
            return Explanation(
                prediction=0.0,
                confidence=0.0,
                feature_importances={},
                explanation_method="lime_unavailable"
            )
        
        try:
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )
            
            # Extract feature importances
            feature_importances = {}
            for feat_name, weight in exp.as_list():
                feature_importances[feat_name] = abs(weight)
            
            # Normalize
            total = sum(feature_importances.values())
            if total > 0:
                feature_importances = {
                    k: v / total for k, v in feature_importances.items()
                }
            
            # Get prediction
            if self.mode == "classification":
                prediction = self.model.predict_proba(instance.reshape(1, -1))[0].max()
            else:
                prediction = self.model.predict(instance.reshape(1, -1))[0]
            
            return Explanation(
                prediction=float(prediction),
                confidence=1.0 - float(exp.score),  # exp.score is error
                feature_importances=feature_importances,
                explanation_method="lime",
                additional_context={"exp_score": exp.score}
            )
        
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return Explanation(
                prediction=0.0,
                confidence=0.0,
                feature_importances={},
                explanation_method="lime_error"
            )


# ============================================================================
# Attention Visualization
# ============================================================================

class AttentionVisualizer:
    """Visualize attention weights for transformer models."""
    
    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        """
        Args:
            model: Transformer or attention-based model
            layer_name: Specific attention layer to visualize
        """
        self.model = model
        self.layer_name = layer_name
        self.attention_weights = None
        self.hook_handle = None
        
        # Register hook to capture attention weights
        self._register_attention_hook()
    
    def _register_attention_hook(self):
        """Register hook to capture attention weights."""
        def hook(module, input, output):
            # Capture attention weights (typically in output)
            if isinstance(output, tuple) and len(output) > 0:
                if hasattr(output[0], 'attention_weights'):
                    self.attention_weights = output[0].attention_weights
                elif isinstance(output[1], torch.Tensor):
                    self.attention_weights = output[1]
        
        # Find and hook the specified layer
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'self_attention' in name.lower():
                if self.layer_name is None or self.layer_name in name:
                    self.hook_handle = module.register_forward_hook(hook)
                    logger.info(f"Registered attention hook on: {name}")
                    break
    
    def visualize(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Visualize attention for input data.
        
        Args:
            input_data: Input tensor to model
            
        Returns:
            visualization: Dictionary with attention data
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        
        if self.attention_weights is None:
            logger.warning("No attention weights captured")
            return {}
        
        # Process attention weights
        if self.attention_weights.dim() == 4:
            # Multi-head attention: (batch, heads, seq_len, seq_len)
            attention_avg = self.attention_weights.mean(dim=1)[0]  # Average over heads
        else:
            attention_avg = self.attention_weights[0]
        
        return {
            "attention_matrix": attention_avg.cpu().numpy(),
            "shape": attention_avg.shape,
            "mean_attention": attention_avg.mean().item(),
            "max_attention": attention_avg.max().item()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from attention weights.
        
        Returns:
            importance: Feature importance dictionary
        """
        if self.attention_weights is None:
            return {}
        
        # Aggregate attention across all positions
        if self.attention_weights.dim() == 4:
            attention_avg = self.attention_weights.mean(dim=(0, 1))  # Avg over batch and heads
        else:
            attention_avg = self.attention_weights.mean(dim=0)
        
        # Feature importance = row sums of attention
        feature_importance = attention_avg.sum(dim=0)
        feature_importance = feature_importance / feature_importance.sum()
        
        return {
            f"feature_{i}": float(feature_importance[i])
            for i in range(len(feature_importance))
        }


# ============================================================================
# Counterfactual Explanations
# ============================================================================

class CounterfactualExplainer:
    """Generate counterfactual explanations for predictions."""
    
    def __init__(self, model, feature_ranges: Dict[str, Tuple[float, float]],
                categorical_features: Optional[List[str]] = None):
        """
        Args:
            model: Prediction function
            feature_ranges: Min/max values for each feature
            categorical_features: Names of categorical features
        """
        self.model = model
        self.feature_ranges = feature_ranges
        self.categorical_features = categorical_features or []
    
    def generate(self, instance: np.ndarray, target_prediction: float,
                max_changes: int = 5) -> Optional[CounterfactualExample]:
        """
        Generate counterfactual example by minimal feature changes.
        
        Args:
            instance: Original instance
            target_prediction: Desired prediction
            max_changes: Maximum feature changes allowed
            
        Returns:
            counterfactual: CounterfactualExample or None
        """
        from scipy.optimize import differential_evolution
        
        n_features = len(instance)
        
        # Create bounds for optimization
        bounds = [
            self.feature_ranges.get(f"feature_{i}", (0, 1))
            for i in range(n_features)
        ]
        
        def objective(x):
            """Loss function: prediction error + sparsity."""
            pred = self.model.predict(x.reshape(1, -1))[0]
            pred_loss = (pred - target_prediction) ** 2
            
            # Sparsity: prefer few changes
            changes = np.abs(x - instance)
            change_loss = changes.sum()
            
            return pred_loss + 0.5 * change_loss
        
        # Find counterfactual
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            seed=42,
            workers=1
        )
        
        if result.fun < 0.1:  # Good solution found
            counterfactual = result.x
            changes = counterfactual - instance
            num_changes = (changes != 0).sum()
            
            if num_changes <= max_changes:
                new_pred = self.model.predict(counterfactual.reshape(1, -1))[0]
                
                return CounterfactualExample(
                    original_instance=instance,
                    counterfactual_instance=counterfactual,
                    changes={
                        f"feature_{i}": float(changes[i])
                        for i in range(len(changes))
                    },
                    num_changes=int(num_changes),
                    new_prediction=float(new_pred),
                    plausibility_score=1.0 - (result.fun / 2)
                )
        
        return None


# ============================================================================
# Feature Interaction Analysis
# ============================================================================

class FeatureInteractionAnalyzer:
    """Analyze feature interactions and dependencies."""
    
    def __init__(self, model, data: np.ndarray):
        """
        Args:
            model: Prediction function
            data: Training data for statistics
        """
        self.model = model
        self.data = data
        self.correlations = np.corrcoef(data.T)
    
    def compute_h_statistic(self, feature_i: int, feature_j: int,
                           samples: int = 100) -> float:
        """
        Compute H-statistic for feature interaction strength.
        
        H-statistic near 0: no interaction
        H-statistic > 0.1: strong interaction
        
        Args:
            feature_i: First feature index
            feature_j: Second feature index
            samples: Number of samples to use
            
        Returns:
            h_stat: H-statistic value
        """
        n_features = self.data.shape[1]
        sample_indices = np.random.choice(len(self.data), samples, replace=True)
        
        h_values = []
        for idx in sample_indices:
            base_x = self.data[idx].copy()
            
            # Prediction with both features
            pred_both = self.model.predict(base_x.reshape(1, -1))[0]
            
            # Prediction with feature_i shuffled
            x_shuffle_i = base_x.copy()
            x_shuffle_i[feature_i] = np.random.choice(self.data[:, feature_i])
            pred_shuffle_i = self.model.predict(x_shuffle_i.reshape(1, -1))[0]
            
            # Prediction with feature_j shuffled
            x_shuffle_j = base_x.copy()
            x_shuffle_j[feature_j] = np.random.choice(self.data[:, feature_j])
            pred_shuffle_j = self.model.predict(x_shuffle_j.reshape(1, -1))[0]
            
            # H-statistic
            h = (pred_both - pred_shuffle_i) - (pred_shuffle_j - pred_both)
            h_values.append(abs(h))
        
        return float(np.mean(h_values))
    
    def analyze_interactions(self, n_top_pairs: int = 10) -> List[Tuple]:
        """
        Analyze all feature interactions.
        
        Args:
            n_top_pairs: Number of top interactions to return
            
        Returns:
            top_interactions: List of (feature_i, feature_j, h_stat) tuples
        """
        n_features = self.data.shape[1]
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                h_stat = self.compute_h_statistic(i, j)
                interactions.append((i, j, h_stat))
        
        # Sort by H-statistic (descending)
        interactions.sort(key=lambda x: x[2], reverse=True)
        
        return interactions[:n_top_pairs]


# ============================================================================
# Unified XAI Controller
# ============================================================================

class XAIController:
    """Unified controller for all XAI methods."""
    
    def __init__(self, model, training_data: np.ndarray,
                feature_names: Optional[List[str]] = None,
                feature_ranges: Optional[Dict] = None):
        """
        Args:
            model: Trained prediction model
            training_data: Training dataset
            feature_names: Feature names
            feature_ranges: Feature min/max ranges
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
        
        # Initialize all explainers
        self.shap_explainer = SHAPExplainer(model, training_data)
        self.lime_explainer = LIMEExplainer(model, training_data, self.feature_names)
        self.counterfactual_explainer = CounterfactualExplainer(
            model, 
            feature_ranges or {f"feature_{i}": (0, 1) for i in range(training_data.shape[1])}
        )
        self.interaction_analyzer = FeatureInteractionAnalyzer(model, training_data)
    
    def explain_prediction(self, instance: np.ndarray,
                          methods: List[str] = None) -> Dict[str, Explanation]:
        """
        Get explanations using multiple methods.
        
        Args:
            instance: Instance to explain
            methods: List of explanation methods to use
            
        Returns:
            explanations: Dictionary of explanations by method
        """
        if methods is None:
            methods = ["shap", "lime"]
        
        explanations = {}
        
        if "shap" in methods:
            explanations["shap"] = self.shap_explainer.explain(instance)
        
        if "lime" in methods:
            explanations["lime"] = self.lime_explainer.explain(instance)
        
        return explanations
    
    def get_feature_importance_summary(self, instance: np.ndarray) -> Dict[str, float]:
        """Get consensus feature importance across methods."""
        exp_shap = self.shap_explainer.explain(instance)
        exp_lime = self.lime_explainer.explain(instance)
        
        # Average importances
        importance = {}
        for fname in self.feature_names:
            shap_imp = exp_shap.feature_importances.get(fname, 0.0)
            lime_imp = exp_lime.feature_importances.get(fname, 0.0)
            importance[fname] = (shap_imp + lime_imp) / 2
        
        return importance
    
    def get_feature_interactions(self, n_top: int = 10) -> List[Tuple]:
        """Get top feature interactions."""
        return self.interaction_analyzer.analyze_interactions(n_top)


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'Explanation',
    'CounterfactualExample',
    'SHAPExplainer',
    'LIMEExplainer',
    'AttentionVisualizer',
    'CounterfactualExplainer',
    'FeatureInteractionAnalyzer',
    'XAIController'
]

if __name__ == "__main__":
    print("Explainable AI Module for AegisPCAP")
    print("Use: from ml.explainable_ai import XAIController")
