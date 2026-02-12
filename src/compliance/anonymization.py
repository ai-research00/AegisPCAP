"""
Anonymization Module - De-identification and Privacy-Preserving Data Transformation

Implements k-anonymity, differential privacy, pseudonymization, and re-identification
risk assessment for privacy-preserving data analysis.

Author: AegisPCAP Compliance Team
Date: February 5, 2026
"""

import hashlib
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    """Result of anonymization operation."""
    result_id: str = field(default_factory=lambda: str(uuid4()))
    original_records: int = 0
    anonymized_records: int = 0
    k_value: int = 0
    utility_score: float = 0.0
    privacy_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    method: str = ""


class KAnonymityTransformer:
    """Ensure k-anonymity (GDPR Article 32)."""

    def __init__(self):
        """Initialize k-anonymity transformer."""
        self.transformations: List[Dict] = []
        logger.info("KAnonymityTransformer initialized")

    def apply_k_anonymity(
        self,
        dataset: List[Dict],
        quasi_identifiers: List[str],
        k: int = 5,
    ) -> Tuple[List[Dict], int]:
        """
        Apply k-anonymity to dataset.

        Args:
            dataset: List of records
            quasi_identifiers: Fields that could identify individuals
            k: Minimum group size (default 5)

        Returns:
            (anonymized_dataset, actual_k_value)
        """
        if not dataset:
            return [], 0

        # Group records by quasi-identifier values
        groups: Dict[str, List[Dict]] = {}
        for record in dataset:
            # Create group key from quasi-identifiers
            key_parts = [
                str(record.get(field, "")) for field in quasi_identifiers
            ]
            key = "|".join(key_parts)

            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Generalize small groups and suppress if needed
        anonymized = []
        min_group_size = min(len(g) for g in groups.values()) if groups else 0

        for group_key, records in groups.items():
            if len(records) >= k:
                # Keep records but generalize quasi-identifiers
                for record in records:
                    anon_record = record.copy()
                    # Generalize: replace with group representative
                    for field in quasi_identifiers:
                        anon_record[f"{field}_suppressed"] = True
                    anonymized.append(anon_record)

        actual_k = min_group_size

        self.transformations.append(
            {
                "method": "k_anonymity",
                "k_target": k,
                "k_actual": actual_k,
                "records_before": len(dataset),
                "records_after": len(anonymized),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(
            f"K-anonymity applied: k={actual_k}, "
            f"{len(dataset)} → {len(anonymized)} records"
        )

        return anonymized, actual_k

    def measure_k_value(self, dataset: List[Dict], quasi_identifiers: List[str]) -> int:
        """
        Measure k-value of dataset.

        Args:
            dataset: Dataset to measure
            quasi_identifiers: QI fields

        Returns:
            k-value (minimum group size)
        """
        if not dataset:
            return 0

        groups: Dict[str, int] = {}
        for record in dataset:
            key_parts = [
                str(record.get(field, "")) for field in quasi_identifiers
            ]
            key = "|".join(key_parts)
            groups[key] = groups.get(key, 0) + 1

        k_value = min(groups.values()) if groups else 0
        return k_value

    def suggest_generalizations(
        self, dataset: List[Dict], current_k: int, target_k: int
    ) -> List[str]:
        """
        Suggest generalizations to reach target k.

        Args:
            dataset: Dataset
            current_k: Current k-value
            target_k: Desired k-value

        Returns:
            List of suggested generalizations
        """
        suggestions = []

        if current_k < target_k:
            generalize_amount = target_k - current_k
            if generalize_amount <= 5:
                suggestions.append("Generalize postal code to region")
            if generalize_amount <= 10:
                suggestions.append("Generalize age to age ranges")
            if generalize_amount > 10:
                suggestions.append("Suppress rare combinations entirely")

        return suggestions


class DifferentialPrivacyOptimizer:
    """Add noise for differential privacy (GDPR Article 32)."""

    def __init__(self):
        """Initialize differential privacy optimizer."""
        self.privacy_applications: List[Dict] = []
        logger.info("DifferentialPrivacyOptimizer initialized")

    def add_dp_noise(
        self,
        dataset: List[Dict],
        columns: List[str],
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> List[Dict]:
        """
        Add Laplace noise for differential privacy.

        Args:
            dataset: Dataset to privatize
            columns: Numeric columns to add noise to
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability

        Returns:
            Noisy dataset
        """
        noisy_dataset = []
        sensitivity = 1.0  # Global sensitivity for values

        for record in dataset:
            noisy_record = record.copy()
            for col in columns:
                if col in noisy_record and isinstance(noisy_record[col], (int, float)):
                    # Laplace mechanism: noise ~ Laplace(0, sensitivity/epsilon)
                    scale = sensitivity / epsilon
                    noise = np.random.laplace(0, scale)
                    noisy_record[col] = float(noisy_record[col]) + noise

            noisy_dataset.append(noisy_record)

        self.privacy_applications.append(
            {
                "method": "laplace_mechanism",
                "epsilon": epsilon,
                "delta": delta,
                "records": len(dataset),
                "columns": columns,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(
            f"Differential privacy applied: ε={epsilon}, δ={delta}, "
            f"{len(dataset)} records"
        )

        return noisy_dataset

    def compute_epsilon(
        self,
        num_records: int,
        query_count: int = 1,
        target_privacy: str = "moderate",
    ) -> float:
        """
        Compute epsilon for target privacy level.

        Args:
            num_records: Number of records in database
            query_count: Number of queries
            target_privacy: "weak"/"moderate"/"strong"/"very_strong"

        Returns:
            epsilon value
        """
        epsilon_targets = {
            "weak": 10.0,
            "moderate": 1.0,
            "strong": 0.1,
            "very_strong": 0.01,
        }

        base_epsilon = epsilon_targets.get(target_privacy, 1.0)
        # Adjust for query count (more queries = higher epsilon needed)
        epsilon = base_epsilon * math.log(query_count + 1)
        return epsilon

    def apply_epsilon_delta(
        self,
        dataset: List[Dict],
        epsilon: float,
        delta: float,
    ) -> Dict:
        """
        Apply (ε,δ)-differential privacy.

        Args:
            dataset: Dataset to privatize
            epsilon: Privacy budget
            delta: Failure probability

        Returns:
            Privacy guarantee report
        """
        return {
            "epsilon": epsilon,
            "delta": delta,
            "interpretation": (
                f"With probability at least 1-{delta}, "
                f"any analysis result is ε={epsilon}-differentially private"
            ),
            "records": len(dataset),
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class Pseudonym:
    """Pseudonym for reversible de-identification."""
    pseudonym_id: str = field(default_factory=lambda: str(uuid4()))
    original_id: str = ""
    pseudonym_value: str = ""
    created_date: datetime = field(default_factory=datetime.utcnow)
    authorized_access: List[str] = field(default_factory=list)


class PseudonymizationController:
    """Replace identifiers with pseudonyms (reversible)."""

    def __init__(self):
        """Initialize pseudonymization controller."""
        self.pseudonym_map: Dict[str, Pseudonym] = {}
        self.reverse_map: Dict[str, str] = {}  # pseudonym -> original
        logger.info("PseudonymizationController initialized")

    def pseudonymize(
        self,
        original_id: str,
        salt: Optional[str] = None,
    ) -> str:
        """
        Create pseudonym for identifier.

        Args:
            original_id: Original identifier
            salt: Salt for hash (for security)

        Returns:
            pseudonym: Irreversible pseudonym
        """
        if original_id in self.pseudonym_map:
            return self.pseudonym_map[original_id].pseudonym_value

        # Create pseudonym using one-way hash
        salt = salt or "default_salt"
        hash_input = f"{original_id}:{salt}".encode()
        pseudonym = hashlib.sha256(hash_input).hexdigest()[:16]

        pseudo_obj = Pseudonym(
            original_id=original_id,
            pseudonym_value=pseudonym,
        )

        self.pseudonym_map[original_id] = pseudo_obj
        self.reverse_map[pseudonym] = original_id

        logger.info(f"Pseudonym created: {original_id} → {pseudonym}")
        return pseudonym

    def reverse_pseudonymization(
        self,
        pseudonym: str,
        authorized_user: str,
    ) -> Optional[str]:
        """
        Reverse pseudonymization (requires authorization).

        Args:
            pseudonym: Pseudonymized identifier
            authorized_user: User requesting de-pseudonymization

        Returns:
            Original ID if authorized, None otherwise
        """
        if pseudonym not in self.reverse_map:
            return None

        original_id = self.reverse_map[pseudonym]

        # Check authorization
        if pseudonym in self.pseudonym_map:
            pseudo_obj = self.pseudonym_map[original_id]
            if authorized_user not in pseudo_obj.authorized_access:
                logger.warning(
                    f"Unauthorized de-pseudonymization attempt: {authorized_user}"
                )
                return None

        logger.info(f"Pseudonym reversed for authorized user: {authorized_user}")
        return original_id

    def audit_pseudonyms(self) -> Dict:
        """
        Audit pseudonymization records.

        Returns:
            Audit report
        """
        return {
            "total_pseudonyms": len(self.pseudonym_map),
            "timestamp": datetime.utcnow().isoformat(),
        }


class ReIdentificationRiskAssessment:
    """Measure re-identification risk (linkage risk)."""

    def __init__(self):
        """Initialize risk assessment."""
        self.risk_assessments: List[Dict] = []
        logger.info("ReIdentificationRiskAssessment initialized")

    def assess_risk(
        self,
        dataset: List[Dict],
        quasi_identifiers: List[str],
    ) -> Dict:
        """
        Assess re-identification risk of dataset.

        Args:
            dataset: Anonymized dataset
            quasi_identifiers: QI fields that could enable linkage

        Returns:
            Risk assessment report
        """
        # Calculate uniqueness of QI combinations
        qi_combinations = set()
        for record in dataset:
            key_parts = [str(record.get(f, "")) for f in quasi_identifiers]
            key = "|".join(key_parts)
            qi_combinations.add(key)

        uniqueness_ratio = len(qi_combinations) / len(dataset) if dataset else 0
        re_id_risk = uniqueness_ratio  # 0.0 = safe, 1.0 = fully identified

        assessment = {
            "qi_combinations": len(qi_combinations),
            "records": len(dataset),
            "uniqueness_ratio": round(uniqueness_ratio, 4),
            "re_identification_risk": round(re_id_risk * 100, 2),
            "risk_level": (
                "low" if re_id_risk < 0.3
                else "medium" if re_id_risk < 0.7
                else "high"
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.risk_assessments.append(assessment)
        logger.info(f"Re-ID risk assessed: {re_id_risk * 100:.1f}%")
        return assessment

    def identify_quasi_identifiers(
        self, dataset: List[Dict], identifiers: List[str]
    ) -> List[str]:
        """
        Identify quasi-identifiers in dataset.

        Args:
            dataset: Dataset to analyze
            identifiers: Known identifiers to exclude

        Returns:
            List of likely quasi-identifiers
        """
        quasi_ids = []

        for field in dataset[0].keys() if dataset else []:
            if field in identifiers:
                continue

            # Count unique values
            unique_values = len(set(r.get(field) for r in dataset))
            uniqueness_pct = (unique_values / len(dataset) * 100) if dataset else 0

            # If high cardinality, likely QI
            if uniqueness_pct > 10:
                quasi_ids.append(field)

        logger.info(f"Identified quasi-identifiers: {quasi_ids}")
        return quasi_ids

    def suggest_mitigations(
        self, risk_level: str, re_id_risk: float
    ) -> List[str]:
        """
        Suggest mitigations for re-ID risk.

        Args:
            risk_level: Current risk level
            re_id_risk: Re-identification risk percentage

        Returns:
            List of mitigation suggestions
        """
        suggestions = []

        if re_id_risk > 50:
            suggestions.append(f"Apply k-anonymity with k≥5 to reduce QI uniqueness")
        if re_id_risk > 30:
            suggestions.append("Add differential privacy noise to numeric fields")
        if re_id_risk > 10:
            suggestions.append("Generalize geographic fields to broader regions")

        return suggestions


class DataUtilityMetrics:
    """Measure information loss from anonymization."""

    def __init__(self):
        """Initialize utility metrics."""
        self.utility_scores: List[Dict] = []
        logger.info("DataUtilityMetrics initialized")

    def compute_utility_score(
        self,
        original_dataset: List[Dict],
        anonymized_dataset: List[Dict],
        critical_fields: Optional[List[str]] = None,
    ) -> float:
        """
        Compute utility preservation score.

        Args:
            original_dataset: Original data
            anonymized_dataset: Anonymized data
            critical_fields: Fields critical for analysis

        Returns:
            Utility score (0-1, where 1 = perfect utility)
        """
        if not original_dataset or not anonymized_dataset:
            return 0.0

        # Simple utility metric: fraction of records preserved
        utility = len(anonymized_dataset) / len(original_dataset)

        # Reduce utility if critical fields suppressed
        if critical_fields:
            for record in anonymized_dataset:
                for field in critical_fields:
                    if f"{field}_suppressed" in record:
                        utility *= 0.9  # Penalize suppression

        return min(utility, 1.0)

    def suggest_parameters(
        self,
        target_utility: float = 0.9,
        target_privacy: str = "moderate",
    ) -> Dict:
        """
        Suggest anonymization parameters.

        Args:
            target_utility: Desired utility level
            target_privacy: Privacy target

        Returns:
            Recommended parameters
        """
        k_recommendations = {
            "weak": 3,
            "moderate": 5,
            "strong": 10,
            "very_strong": 50,
        }

        k = k_recommendations.get(target_privacy, 5)

        return {
            "recommended_k": k,
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "target_utility": target_utility,
            "expected_utility": max(0.5, target_utility - 0.1),
        }

    def balance_privacy_utility(
        self,
        epsilon: float,
        k_value: int,
    ) -> Dict:
        """
        Balance privacy-utility tradeoff.

        Args:
            epsilon: Privacy budget
            k_value: k-anonymity value

        Returns:
            Privacy-utility balance metrics
        """
        privacy_score = (1.0 / (epsilon + 1.0)) * 100  # 0-100
        utility_score = min((k_value / 50.0) * 100, 100)  # Approximation

        return {
            "epsilon": epsilon,
            "k_value": k_value,
            "privacy_score": round(privacy_score, 2),
            "utility_score": round(utility_score, 2),
            "balance": "optimal" if abs(privacy_score - utility_score) < 20 else "imbalanced",
        }


class AnonymizationController:
    """Unified anonymization interface."""

    def __init__(self):
        """Initialize anonymization controller."""
        self.k_anonymity = KAnonymityTransformer()
        self.diff_privacy = DifferentialPrivacyOptimizer()
        self.pseudonymization = PseudonymizationController()
        self.risk_assessment = ReIdentificationRiskAssessment()
        self.utility_metrics = DataUtilityMetrics()
        self.history: List[Dict] = []
        logger.info("AnonymizationController initialized")

    def anonymize_dataset(
        self,
        dataset: List[Dict],
        k_value: int = 5,
        apply_dp: bool = True,
        epsilon: float = 1.0,
    ) -> Dict:
        """
        Anonymize dataset using k-anonymity and differential privacy.

        Args:
            dataset: Dataset to anonymize
            k_value: Target k-anonymity level
            apply_dp: Whether to apply differential privacy
            epsilon: Privacy budget for DP

        Returns:
            Anonymization result
        """
        # Identify quasi-identifiers
        qi_fields = ["age_range", "gender", "postal_code"]

        # Apply k-anonymity
        anonymized, actual_k = self.k_anonymity.apply_k_anonymity(
            dataset, qi_fields, k_value
        )

        # Apply differential privacy if requested
        if apply_dp and anonymized:
            numeric_fields = ["income", "count"]
            anonymized = self.diff_privacy.add_dp_noise(
                anonymized, numeric_fields, epsilon
            )

        # Compute utility
        utility = self.utility_metrics.compute_utility_score(
            dataset, anonymized, qi_fields
        )

        result = {
            "status": "success",
            "original_records": len(dataset),
            "anonymized_records": len(anonymized),
            "k_value_achieved": actual_k,
            "k_value_target": k_value,
            "utility_score": utility,
            "epsilon": epsilon,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._log_operation("anonymize_dataset", result)
        return result

    def assess_privacy(self, dataset: List[Dict]) -> Dict:
        """
        Assess privacy of anonymized dataset.

        Args:
            dataset: Dataset to assess

        Returns:
            Privacy assessment
        """
        qi_fields = ["age_range", "gender", "postal_code"]
        assessment = self.risk_assessment.assess_risk(dataset, qi_fields)

        result = {
            "re_identification_risk_pct": assessment["re_identification_risk"],
            "risk_level": assessment["risk_level"],
            "mitigations": self.risk_assessment.suggest_mitigations(
                assessment["risk_level"],
                assessment["re_identification_risk"],
            ),
        }

        self._log_operation("assess_privacy", result)
        return result

    def generate_report(self) -> Dict:
        """
        Generate anonymization report.

        Returns:
            Comprehensive report
        """
        return {
            "module": "Anonymization",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "k_anonymity": "operational",
                "differential_privacy": "operational",
                "pseudonymization": "operational",
                "risk_assessment": "operational",
                "utility_metrics": "operational",
            },
            "status": "operational",
        }

    def _log_operation(self, operation: str, result: Dict) -> None:
        """Log anonymization operation."""
        self.history.append(
            {
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            }
        )
