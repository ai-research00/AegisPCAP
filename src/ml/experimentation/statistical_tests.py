"""Statistical tests and analysis for A/B testing.

Provides methods for computing statistical significance, power analysis,
sample size calculation, and confidence intervals.

Functions:
    t_test: Welch's t-test for independent samples
    chi_square: Chi-square test for categorical data
    calculate_sample_size: Calculate required sample size
    calculate_statistical_power: Calculate statistical power
"""

from typing import Dict, Tuple

import numpy as np
from scipy import stats as scipy_stats


def t_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Welch's t-test for independent samples.
    
    Tests if means of two groups are significantly different.
    Assumes unequal variances (Welch's variant).
    
    Args:
        control_values: Control group values
        treatment_values: Treatment group values
        alpha: Significance level
        
    Returns:
        Dictionary with t-statistic, p-value, significance
        
    Example:
        >>> control = np.array([1.0, 2.0, 3.0])
        >>> treatment = np.array([2.0, 3.0, 4.0])
        >>> result = t_test(control, treatment)
        >>> print(result['p_value'])
    """
    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = scipy_stats.ttest_ind(
        control_values,
        treatment_values,
        equal_var=False,
    )
    
    # Cohen's d effect size
    n1, n2 = len(control_values), len(treatment_values)
    var1, var2 = np.var(control_values, ddof=1), np.var(treatment_values, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / (pooled_std + 1e-6)
    
    # Confidence interval for difference
    mean_diff = np.mean(treatment_values) - np.mean(control_values)
    se_diff = np.sqrt(var1/n1 + var2/n2)
    t_crit = scipy_stats.t.ppf(1 - alpha/2, n1+n2-2)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": float(p_value) < alpha,
        "cohens_d": float(cohens_d),
        "mean_difference": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "control_mean": float(np.mean(control_values)),
        "treatment_mean": float(np.mean(treatment_values)),
    }


def chi_square(
    observed: np.ndarray,
    expected: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Chi-square test for categorical data.
    
    Tests if observed frequencies differ from expected.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        alpha: Significance level
        
    Returns:
        Dictionary with chi-square statistic, p-value, significance
    """
    chi2_stat, p_value = scipy_stats.chisquare(observed, expected)
    
    # Cramér's V effect size
    n = np.sum(observed)
    min_dim = min(len(observed) - 1, len(expected) - 1)
    cramers_v = np.sqrt(chi2_stat / (n * (min_dim + 1)))
    
    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "significant": float(p_value) < alpha,
        "cramers_v": float(cramers_v),
    }


def calculate_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Calculate required sample size for A/B test.
    
    Uses normal approximation to binomial distribution.
    
    Args:
        baseline_conversion: Baseline conversion rate (0-1)
        minimum_detectable_effect: Minimum relative lift (0-1)
        alpha: Type I error rate (significance level)
        power: Statistical power (1 - beta)
        
    Returns:
        Required sample size per variant
        
    Example:
        >>> # Detect 5% lift from 10% baseline with 80% power
        >>> n = calculate_sample_size(0.10, 0.05, power=0.80)
        >>> print(f"Need {n} samples per variant")
    """
    # Baseline and treatment conversion rates
    p0 = baseline_conversion
    p1 = baseline_conversion * (1 + minimum_detectable_effect)
    
    # Ensure p1 is valid
    p1 = min(p1, 0.999)
    
    # Z-scores
    z_alpha = scipy_stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = scipy_stats.norm.ppf(power)
    
    # Pooled proportion
    p_pooled = (p0 + p1) / 2
    
    # Sample size formula
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    denominator = (p1 - p0) ** 2
    
    n = int(np.ceil(numerator / denominator))
    
    return max(n, 100)  # Minimum 100 samples


def calculate_statistical_power(
    baseline_conversion: float,
    minimum_detectable_effect: float,
    sample_size: int,
    alpha: float = 0.05,
) -> float:
    """Calculate statistical power for given sample size.
    
    Args:
        baseline_conversion: Baseline conversion rate
        minimum_detectable_effect: Minimum detectable effect
        sample_size: Sample size per variant
        alpha: Significance level
        
    Returns:
        Statistical power (0-1)
    """
    p0 = baseline_conversion
    p1 = baseline_conversion * (1 + minimum_detectable_effect)
    p1 = min(p1, 0.999)
    
    z_alpha = scipy_stats.norm.ppf(1 - alpha/2)
    p_pooled = (p0 + p1) / 2
    
    # Standard error
    se = np.sqrt(2 * p_pooled * (1 - p_pooled) / sample_size)
    
    # Z-score for effect
    z_effect = (p1 - p0) / (se + 1e-6)
    
    # Power = P(Z > z_alpha - z_effect)
    power = scipy_stats.norm.cdf(z_effect - z_alpha)
    
    return float(np.clip(power, 0, 1))


def confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for mean.
    
    Args:
        values: Sample values
        confidence: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(values)
    mean = np.mean(values)
    se = scipy_stats.sem(values)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * se
    
    return float(mean - margin), float(mean + margin)


def relative_lift(
    control_mean: float,
    treatment_mean: float,
) -> float:
    """Calculate relative lift as percentage.
    
    Args:
        control_mean: Control group mean
        treatment_mean: Treatment group mean
        
    Returns:
        Relative lift as percentage
        
    Example:
        >>> lift = relative_lift(100, 105)
        >>> print(f"Lift: {lift:.1f}%")  # 5.0%
    """
    return ((treatment_mean - control_mean) / (control_mean + 1e-6)) * 100


def is_statistically_significant(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Check if difference is statistically significant.
    
    Args:
        control_values: Control group values
        treatment_values: Treatment group values
        alpha: Significance level
        
    Returns:
        True if significant, False otherwise
    """
    result = t_test(control_values, treatment_values, alpha)
    return bool(result["significant"])
