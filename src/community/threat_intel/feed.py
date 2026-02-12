"""
Threat Intelligence Feed

Manages community-driven threat intelligence sharing including
indicator publishing, consumption, validation, and feedback.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of threat indicators."""
    IP = "ip"
    DOMAIN = "domain"
    HASH = "hash"
    URL = "url"
    CERTIFICATE = "certificate"
    EMAIL = "email"


class ValidationStatus(Enum):
    """Validation status for indicators."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ThreatIndicator:
    """Represents a threat indicator."""
    indicator_id: str
    indicator_type: IndicatorType
    value: str
    threat_type: str  # malware, phishing, c2, etc.
    confidence_score: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    false_positive_reports: int = 0
    validation_status: ValidationStatus = ValidationStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert indicator to dictionary."""
        return {
            "indicator_id": self.indicator_id,
            "indicator_type": self.indicator_type.value,
            "value": self.value,
            "threat_type": self.threat_type,
            "confidence_score": self.confidence_score,
            "source": self.source,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "tags": self.tags,
            "context": self.context,
            "false_positive_reports": self.false_positive_reports,
            "validation_status": self.validation_status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreatIndicator":
        """Create indicator from dictionary."""
        data = data.copy()
        data["indicator_type"] = IndicatorType(data["indicator_type"])
        data["validation_status"] = ValidationStatus(data["validation_status"])
        data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**data)


@dataclass
class FeedFilters:
    """Filters for querying threat indicators."""
    indicator_type: Optional[IndicatorType] = None
    threat_type: Optional[str] = None
    min_confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    since: Optional[datetime] = None


@dataclass
class Feedback:
    """Feedback on a threat indicator."""
    feedback_id: str
    indicator_id: str
    user_id: str
    is_false_positive: bool
    comment: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class ThreatIntelligenceFeed:
    """
    Manages threat intelligence sharing.
    
    Provides:
    - Indicator publishing and consumption
    - Validation and quality scoring
    - False positive reporting
    - Indicator expiration
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize threat intelligence feed.
        
        Args:
            storage_dir: Directory for indicator storage (default: ./threat_intel)
        """
        self.storage_dir = storage_dir or Path("./threat_intel")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.indicators_dir = self.storage_dir / "indicators"
        self.indicators_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir = self.storage_dir / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Expiration settings
        self.expiration_days = {
            IndicatorType.IP: 30,
            IndicatorType.DOMAIN: 90,
            IndicatorType.HASH: 365,
            IndicatorType.URL: 60,
            IndicatorType.CERTIFICATE: 180,
            IndicatorType.EMAIL: 90,
        }
        
        logger.info(f"ThreatIntelligenceFeed initialized with storage_dir: {self.storage_dir}")
    
    def publish_indicator(self, indicator: ThreatIndicator, source: str) -> str:
        """
        Publish threat indicator to feed.
        
        Args:
            indicator: Threat indicator to publish
            source: Source of the indicator
            
        Returns:
            Indicator ID
            
        Raises:
            ValueError: If indicator validation fails
        """
        try:
            # Validate indicator
            validation_result = self.validate_indicator(indicator)
            if not validation_result["valid"]:
                raise ValueError(f"Indicator validation failed: {validation_result['errors']}")
            
            # Set source
            indicator.source = source
            
            # Calculate confidence score
            indicator.confidence_score = self._calculate_confidence(indicator)
            
            # Save indicator
            indicator_file = self.indicators_dir / f"{indicator.indicator_id}.json"
            with open(indicator_file, 'w') as f:
                json.dump(indicator.to_dict(), f, indent=2)
            
            logger.info(
                f"Published indicator: {indicator.indicator_type.value} "
                f"{indicator.value} from {source}"
            )
            
            return indicator.indicator_id
            
        except Exception as e:
            logger.error(f"Failed to publish indicator: {e}")
            raise
    
    def consume_indicators(self, filters: FeedFilters) -> List[ThreatIndicator]:
        """
        Retrieve threat indicators matching filters.
        
        Args:
            filters: Filters to apply
            
        Returns:
            List of matching indicators
        """
        indicators = []
        
        # Load all indicators
        for indicator_file in self.indicators_dir.glob("*.json"):
            try:
                with open(indicator_file, 'r') as f:
                    indicator_dict = json.load(f)
                indicator = ThreatIndicator.from_dict(indicator_dict)
                
                # Check expiration
                if self._is_expired(indicator):
                    indicator.validation_status = ValidationStatus.EXPIRED
                    self._save_indicator(indicator)
                    continue
                
                # Apply filters
                if filters.indicator_type and indicator.indicator_type != filters.indicator_type:
                    continue
                
                if filters.threat_type and indicator.threat_type != filters.threat_type:
                    continue
                
                if filters.min_confidence and indicator.confidence_score < filters.min_confidence:
                    continue
                
                if filters.source and indicator.source != filters.source:
                    continue
                
                if filters.tags:
                    if not any(tag in indicator.tags for tag in filters.tags):
                        continue
                
                if filters.since and indicator.last_seen < filters.since:
                    continue
                
                indicators.append(indicator)
                
            except Exception as e:
                logger.warning(f"Failed to load indicator {indicator_file}: {e}")
                continue
        
        logger.info(f"Consumed {len(indicators)} indicators with filters")
        return indicators
    
    def validate_indicator(self, indicator: ThreatIndicator) -> Dict[str, Any]:
        """
        Validate indicator format and quality.
        
        Args:
            indicator: Indicator to validate
            
        Returns:
            Validation result with status and errors
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not indicator.value:
            errors.append("Indicator value is required")
        
        if not indicator.threat_type:
            errors.append("Threat type is required")
        
        if not 0.0 <= indicator.confidence_score <= 1.0:
            errors.append("Confidence score must be between 0.0 and 1.0")
        
        # Type-specific validation
        if indicator.indicator_type == IndicatorType.IP:
            if not self._validate_ip(indicator.value):
                errors.append(f"Invalid IP address: {indicator.value}")
        
        elif indicator.indicator_type == IndicatorType.DOMAIN:
            if not self._validate_domain(indicator.value):
                errors.append(f"Invalid domain: {indicator.value}")
        
        elif indicator.indicator_type == IndicatorType.HASH:
            if not self._validate_hash(indicator.value):
                errors.append(f"Invalid hash: {indicator.value}")
        
        # Check for duplicates
        if self._is_duplicate(indicator):
            warnings.append("Indicator already exists in feed")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def report_false_positive(self, indicator_id: str, feedback: Feedback) -> None:
        """
        Report false positive for indicator.
        
        Args:
            indicator_id: ID of indicator to report
            feedback: Feedback details
        """
        try:
            # Load indicator
            indicator = self._load_indicator(indicator_id)
            if not indicator:
                raise ValueError(f"Indicator not found: {indicator_id}")
            
            # Increment false positive count
            indicator.false_positive_reports += 1
            
            # Adjust confidence score
            indicator.confidence_score = self._calculate_confidence(indicator)
            
            # Save updated indicator
            self._save_indicator(indicator)
            
            # Save feedback
            feedback_file = self.feedback_dir / f"{feedback.feedback_id}.json"
            with open(feedback_file, 'w') as f:
                json.dump({
                    "feedback_id": feedback.feedback_id,
                    "indicator_id": feedback.indicator_id,
                    "user_id": feedback.user_id,
                    "is_false_positive": feedback.is_false_positive,
                    "comment": feedback.comment,
                    "created_at": feedback.created_at.isoformat(),
                }, f, indent=2)
            
            logger.info(f"Recorded false positive report for indicator {indicator_id}")
            
        except Exception as e:
            logger.error(f"Failed to report false positive: {e}")
            raise
    
    def _calculate_confidence(self, indicator: ThreatIndicator) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            indicator: Indicator to score
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = indicator.confidence_score
        
        # Adjust for source reputation (simplified)
        source_reputation = 0.8  # Would be looked up from source registry
        confidence *= source_reputation
        
        # Adjust for age
        age_days = (datetime.utcnow() - indicator.first_seen).days
        age_factor = max(0.5, 1.0 - (age_days / 365))
        confidence *= age_factor
        
        # Adjust for false positive reports
        if indicator.false_positive_reports > 0:
            fp_penalty = 0.1 * indicator.false_positive_reports
            confidence = max(0.1, confidence - fp_penalty)
        
        return min(1.0, confidence)
    
    def _is_expired(self, indicator: ThreatIndicator) -> bool:
        """Check if indicator has expired."""
        expiration_days = self.expiration_days.get(indicator.indicator_type, 90)
        expiration_date = indicator.last_seen + timedelta(days=expiration_days)
        return datetime.utcnow() > expiration_date
    
    def _is_duplicate(self, indicator: ThreatIndicator) -> bool:
        """Check if indicator already exists."""
        for existing_file in self.indicators_dir.glob("*.json"):
            try:
                with open(existing_file, 'r') as f:
                    existing_dict = json.load(f)
                if (existing_dict["indicator_type"] == indicator.indicator_type.value and
                    existing_dict["value"] == indicator.value):
                    return True
            except Exception:
                continue
        return False
    
    def _load_indicator(self, indicator_id: str) -> Optional[ThreatIndicator]:
        """Load indicator by ID."""
        indicator_file = self.indicators_dir / f"{indicator_id}.json"
        if not indicator_file.exists():
            return None
        
        with open(indicator_file, 'r') as f:
            indicator_dict = json.load(f)
        return ThreatIndicator.from_dict(indicator_dict)
    
    def _save_indicator(self, indicator: ThreatIndicator) -> None:
        """Save indicator to storage."""
        indicator_file = self.indicators_dir / f"{indicator.indicator_id}.json"
        with open(indicator_file, 'w') as f:
            json.dump(indicator.to_dict(), f, indent=2)
    
    def _validate_ip(self, value: str) -> bool:
        """Validate IP address format."""
        import ipaddress
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    def _validate_domain(self, value: str) -> bool:
        """Validate domain format."""
        import re
        pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    def _validate_hash(self, value: str) -> bool:
        """Validate hash format (MD5, SHA1, SHA256)."""
        import re
        # MD5: 32 hex, SHA1: 40 hex, SHA256: 64 hex
        pattern = r'^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$'
        return bool(re.match(pattern, value))
