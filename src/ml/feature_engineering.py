"""
Advanced Feature Engineering for Phase 10: Advanced Analytics

Implements:
- Genetic Programming for automated feature discovery
- Feature interaction detection and selection
- Domain-specific feature engineering for network traffic
- Feature importance and correlation analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Optional, Dict, Set, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import sympy
from itertools import combinations
import pandas as pd

logger = logging.getLogger(__name__)


# ===== DATA STRUCTURES =====

@dataclass
class FeatureEngineringConfig:
    """Configuration for feature engineering."""
    
    # Genetic programming
    population_size: int = 100
    num_generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    tournament_size: int = 5
    
    # Feature interactions
    max_interaction_degree: int = 3
    interaction_threshold: float = 0.1
    select_top_k: int = 20
    
    # Domain-specific
    packet_features: List[str] = None
    flow_features: List[str] = None
    protocol_features: List[str] = None


# ===== GENETIC PROGRAMMING =====

class Individual:
    """Represents a feature transformation program."""
    
    def __init__(self, expression: str, fitness: float = None):
        """
        Initialize individual.
        
        Args:
            expression: Mathematical expression using features
            fitness: Fitness score (lower is better)
        """
        self.expression = expression
        self.fitness = fitness
        self.code = self._compile_expression(expression)
    
    @staticmethod
    def _compile_expression(expr: str) -> Callable:
        """Compile expression to callable function."""
        try:
            # Create lambda function
            func = eval(f'lambda x: {expr}')
            return func
        except:
            return lambda x: np.zeros(len(x))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate fitness on data.
        
        Args:
            X: Feature matrix (N, D)
            y: Target labels (N,)
        
        Returns:
            Fitness score (MSE for regression, -accuracy for classification)
        """
        try:
            # Evaluate expression on each sample
            predictions = np.array([
                self._safe_eval(sample)
                for sample in X
            ])
            
            # Handle invalid predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return float('inf')
            
            # Compute fitness
            if predictions.ndim == 1:
                mse = np.mean((predictions - y) ** 2)
            else:
                # Classification: accuracy
                pred_labels = predictions.argmax(axis=1)
                mse = -np.mean(pred_labels == y)
            
            self.fitness = mse
            return mse
        except:
            self.fitness = float('inf')
            return float('inf')
    
    def _safe_eval(self, sample: np.ndarray) -> float:
        """Safely evaluate expression on single sample."""
        try:
            # Create variable namespace
            ns = {f'x{i}': val for i, val in enumerate(sample)}
            ns.update({
                'np': np,
                'abs': abs,
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-8),
                'sqrt': lambda x: np.sqrt(np.abs(x))
            })
            
            result = eval(self.expression, ns)
            return float(result) if isinstance(result, (int, float, np.number)) else result
        except:
            return 0.0
    
    def mutate(self, mutation_rate: float = 0.2) -> 'Individual':
        """
        Create mutated copy of individual.
        
        Args:
            mutation_rate: Probability of mutation
        
        Returns:
            New mutated individual
        """
        tokens = self.expression.split()
        
        for i in range(len(tokens)):
            if np.random.random() < mutation_rate:
                token = tokens[i]
                
                # Replace with random token
                if token.startswith('x'):
                    # Replace feature reference
                    tokens[i] = f'x{np.random.randint(0, 10)}'
                elif token in ['+', '-', '*', '/']:
                    # Replace operator
                    tokens[i] = np.random.choice(['+', '-', '*', '/'])
                elif token in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    # Replace function
                    tokens[i] = np.random.choice(['sin', 'cos', 'exp', 'log', 'sqrt'])
        
        new_expr = ' '.join(tokens)
        return Individual(new_expr)
    
    @staticmethod
    def crossover(parent1: 'Individual', parent2: 'Individual') -> 'Individual':
        """
        Create offspring through crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Offspring individual
        """
        tokens1 = parent1.expression.split()
        tokens2 = parent2.expression.split()
        
        # Random crossover point
        point1 = np.random.randint(1, len(tokens1))
        point2 = np.random.randint(1, len(tokens2))
        
        # Combine tokens
        offspring_tokens = tokens1[:point1] + tokens2[point2:]
        offspring_expr = ' '.join(offspring_tokens)
        
        return Individual(offspring_expr)


class GeneticProgrammingFeatureGenerator:
    """
    Generates new features using genetic programming.
    
    Evolves mathematical expressions to create informative features.
    """
    
    def __init__(self, config: FeatureEngineringConfig):
        """Initialize genetic programming feature generator."""
        self.config = config
        self.population = []
        self.best_individual = None
        self.history = []
    
    def initialize_population(self, num_features: int):
        """Initialize random population."""
        operators = ['+', '-', '*', '/']
        functions = ['sin', 'cos', 'exp', 'log', 'sqrt']
        
        for _ in range(self.config.population_size):
            # Create random expression
            tokens = []
            
            for _ in range(np.random.randint(3, 8)):
                if np.random.random() < 0.3:
                    # Feature reference
                    tokens.append(f'x{np.random.randint(0, num_features)}')
                elif np.random.random() < 0.5:
                    # Operator
                    tokens.append(np.random.choice(operators))
                else:
                    # Function
                    tokens.append(np.random.choice(functions))
            
            expr = ' '.join(tokens)
            individual = Individual(expr)
            self.population.append(individual)
    
    def evolve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_generations: int = None
    ) -> List[Individual]:
        """
        Evolve population for specified generations.
        
        Args:
            X: Feature matrix (N, D)
            y: Target (N,)
            num_generations: Number of generations
        
        Returns:
            Best individuals from final generation
        """
        num_generations = num_generations or self.config.num_generations
        
        for generation in range(num_generations):
            # Evaluate fitness
            for individual in self.population:
                individual.evaluate(X, y)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness)
            
            # Track best
            if self.best_individual is None or self.population[0].fitness < self.best_individual.fitness:
                self.best_individual = self.population[0]
            
            self.history.append(self.best_individual.fitness)
            
            logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.4f}")
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep top individuals
            elite_size = max(1, self.config.population_size // 10)
            new_population.extend(self.population[:elite_size])
            
            # Tournament selection + crossover + mutation
            while len(new_population) < self.config.population_size:
                # Tournament selection
                tournament = np.random.choice(
                    self.population,
                    size=self.config.tournament_size,
                    replace=False
                )
                parent1 = min(tournament, key=lambda x: x.fitness)
                
                tournament = np.random.choice(
                    self.population,
                    size=self.config.tournament_size,
                    replace=False
                )
                parent2 = min(tournament, key=lambda x: x.fitness)
                
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    offspring = Individual.crossover(parent1, parent2)
                else:
                    offspring = parent1
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    offspring = offspring.mutate(self.config.mutation_rate)
                
                new_population.append(offspring)
            
            self.population = new_population[:self.config.population_size]
        
        # Return top individuals
        self.population.sort(key=lambda x: x.fitness)
        return self.population[:self.config.population_size // 10]
    
    def get_best_features(self, k: int = 5) -> List[str]:
        """Get top-k evolved feature expressions."""
        self.population.sort(key=lambda x: x.fitness)
        return [ind.expression for ind in self.population[:k]]


# ===== FEATURE INTERACTIONS =====

class InteractionDetector:
    """
    Detects and selects important feature interactions.
    
    Methods:
    - Statistical dependency tests
    - Information gain analysis
    - Mutual information computation
    """
    
    def __init__(self, config: FeatureEngineringConfig):
        """Initialize interaction detector."""
        self.config = config
        self.interaction_scores = {}
    
    def find_interactions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Find important feature interactions.
        
        Args:
            X: Feature matrix (N, D)
            y: Target (N,)
            feature_names: Optional feature names
        
        Returns:
            List of (interaction indices, importance score) sorted by importance
        """
        n_features = X.shape[1]
        interactions = []
        
        # Compute pairwise interactions (degree 2)
        logger.info("Computing pairwise interactions...")
        for i, j in combinations(range(n_features), 2):
            interaction = X[:, i] * X[:, j]
            score = self._compute_importance(interaction, y)
            
            if score > self.config.interaction_threshold:
                interactions.append(((i, j), score))
        
        # Compute 3-way interactions if applicable
        if self.config.max_interaction_degree >= 3 and n_features <= 20:
            logger.info("Computing 3-way interactions...")
            for i, j, k in combinations(range(n_features), 3):
                interaction = X[:, i] * X[:, j] * X[:, k]
                score = self._compute_importance(interaction, y)
                
                if score > self.config.interaction_threshold:
                    interactions.append(((i, j, k), score))
        
        # Sort by importance
        interactions.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        interactions = interactions[:self.config.select_top_k]
        
        return interactions
    
    @staticmethod
    def _compute_importance(feature: np.ndarray, y: np.ndarray) -> float:
        """Compute feature importance using mutual information."""
        # Normalize feature
        feature = (feature - feature.mean()) / (feature.std() + 1e-8)
        
        # Discretize into bins
        feature_binned = np.digitize(feature, np.percentile(feature, np.linspace(0, 100, 11)))
        y_binned = np.digitize(y, np.percentile(y, np.linspace(0, 100, 11)))
        
        # Mutual information
        mi = InteractionDetector._mutual_information(feature_binned, y_binned)
        
        return mi
    
    @staticmethod
    def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two discrete variables."""
        # Joint distribution
        p_xy = np.zeros((x.max() + 1, y.max() + 1))
        for i, j in zip(x, y):
            p_xy[i, j] += 1
        p_xy /= p_xy.sum()
        
        # Marginal distributions
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Mutual information
        mi = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-10) + 1e-10)
        
        return max(0, mi)


# ===== DOMAIN-SPECIFIC FEATURES =====

class NetworkTrafficFeatureEngineer:
    """
    Domain-specific feature engineering for network traffic analysis.
    
    Creates features from packet flows, protocols, and statistics.
    """
    
    @staticmethod
    def compute_flow_statistics(
        packets: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute statistical features from packet sequences.
        
        Args:
            packets: Packet features (N, max_packets, features_per_packet)
            labels: Optional packet labels
        
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        # Packet size statistics
        features['packet_size_mean'] = np.mean(packets[:, :, 0], axis=1)
        features['packet_size_std'] = np.std(packets[:, :, 0], axis=1)
        features['packet_size_max'] = np.max(packets[:, :, 0], axis=1)
        features['packet_size_min'] = np.min(packets[:, :, 0], axis=1)
        
        # Packet rate (inter-arrival times)
        if packets.shape[-1] > 1:
            inter_arrivals = np.diff(packets[:, :, 1], axis=1)
            features['inter_arrival_mean'] = np.mean(inter_arrivals, axis=1)
            features['inter_arrival_std'] = np.std(inter_arrivals, axis=1)
        
        # Packet count
        features['total_packets'] = np.sum(packets[:, :, 0] > 0, axis=1)
        
        # Direction changes (if bidirectional)
        if packets.shape[-1] > 2:
            directions = packets[:, :, 2]  # Assume direction in 3rd feature
            features['direction_changes'] = np.sum(np.diff(directions, axis=1) != 0, axis=1)
        
        return features
    
    @staticmethod
    def compute_protocol_features(
        protocols: np.ndarray,
        packet_counts: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute features from protocol distribution.
        
        Args:
            protocols: Protocol types per packet (N, max_packets)
            packet_counts: Number of packets in each flow (N,)
        
        Returns:
            Dictionary of protocol features
        """
        features = {}
        
        unique_protocols = np.unique(protocols)
        
        # Protocol diversity
        features['protocol_diversity'] = np.array([
            len(np.unique(protocols[i, :packet_counts[i]]))
            for i in range(len(packet_counts))
        ])
        
        # Protocol distribution
        for proto in unique_protocols[:5]:  # Top 5 protocols
            features[f'protocol_{proto}_ratio'] = np.array([
                np.sum(protocols[i, :packet_counts[i]] == proto) / max(1, packet_counts[i])
                for i in range(len(packet_counts))
            ])
        
        return features
    
    @staticmethod
    def compute_anomaly_indicators(
        features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute features indicating anomalous behavior.
        
        Args:
            features: Dictionary of computed features
        
        Returns:
            Anomaly indicator features
        """
        anomaly_features = {}
        
        # Extreme packet sizes
        if 'packet_size_max' in features:
            mean = np.mean(features['packet_size_max'])
            std = np.std(features['packet_size_max'])
            anomaly_features['oversized_packets'] = (
                (features['packet_size_max'] - mean) / (std + 1e-8)
            )
        
        # Unusual inter-arrival patterns
        if 'inter_arrival_std' in features:
            anomaly_features['irregular_timing'] = features['inter_arrival_std']
        
        # Protocol anomalies
        if 'protocol_diversity' in features:
            diversity_mean = np.mean(features['protocol_diversity'])
            anomaly_features['unusual_protocol_mix'] = (
                features['protocol_diversity'] / diversity_mean
            )
        
        return anomaly_features


# ===== FEATURE SELECTION =====

class FeatureSelector:
    """
    Selects most important features using various criteria.
    
    Methods:
    - Mutual information
    - Correlation analysis
    - Permutation importance
    - Model-based importance
    """
    
    @staticmethod
    def mutual_information_ranking(
        X: np.ndarray,
        y: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rank features by mutual information with target.
        
        Args:
            X: Feature matrix (N, D)
            y: Target (N,)
            k: Number of features to select (None = all)
        
        Returns:
            Tuple of (feature indices, importance scores)
        """
        scores = []
        
        for i in range(X.shape[1]):
            feature = X[:, i]
            feature_binned = np.digitize(
                feature,
                np.percentile(feature, np.linspace(0, 100, 11))
            )
            
            if y.dtype in [np.int32, np.int64]:
                y_binned = y
            else:
                y_binned = np.digitize(
                    y,
                    np.percentile(y, np.linspace(0, 100, 11))
                )
            
            mi = InteractionDetector._mutual_information(feature_binned, y_binned)
            scores.append(mi)
        
        # Sort by importance
        indices = np.argsort(scores)[::-1]
        scores_sorted = [scores[i] for i in indices]
        
        if k is not None:
            indices = indices[:k]
            scores_sorted = scores_sorted[:k]
        
        return list(indices), scores_sorted
    
    @staticmethod
    def correlation_filtering(
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.1,
        k: Optional[int] = None
    ) -> List[int]:
        """
        Select features with high correlation with target.
        
        Args:
            X: Feature matrix (N, D)
            y: Target (N,)
            threshold: Minimum correlation threshold
            k: Number of features to select
        
        Returns:
            List of selected feature indices
        """
        correlations = []
        
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if np.isnan(corr):
                corr = 0
            correlations.append(abs(corr))
        
        # Filter by threshold
        selected = [i for i, c in enumerate(correlations) if c > threshold]
        
        # Sort by correlation
        selected.sort(key=lambda i: correlations[i], reverse=True)
        
        # Limit to k features
        if k is not None:
            selected = selected[:k]
        
        return selected


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Create synthetic dataset
    X = np.random.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(100) * 0.1
    
    # Genetic programming
    config = FeatureEngineringConfig()
    gp = GeneticProgrammingFeatureGenerator(config)
    gp.initialize_population(X.shape[1])
    best_features = gp.evolve(X, y, num_generations=10)
    
    print("Top features discovered by genetic programming:")
    for feat in gp.get_best_features(3):
        print(f"  {feat}")
    
    # Feature interactions
    detector = InteractionDetector(config)
    interactions = detector.find_interactions(X, y)
    print(f"\nTop interactions found: {interactions[:3]}")
    
    # Feature selection
    indices, scores = FeatureSelector.mutual_information_ranking(X, y, k=3)
    print(f"\nTop 3 features by MI: {indices} (scores: {scores})")
