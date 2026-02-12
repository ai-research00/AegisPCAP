"""
STIX Converter

Converts between AegisPCAP internal format and STIX 2.1 format
for threat intelligence exchange.
"""

import logging
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from .feed import ThreatIndicator, IndicatorType, ValidationStatus

logger = logging.getLogger(__name__)


class STIXConverter:
    """
    Converts between STIX/TAXII and internal format.
    
    Supports STIX 2.1 specification for threat intelligence exchange.
    """
    
    def __init__(self):
        """Initialize STIX converter."""
        self.stix_version = "2.1"
        logger.info("STIXConverter initialized (STIX 2.1)")
    
    def to_stix(self, indicator: ThreatIndicator) -> Dict[str, Any]:
        """
        Convert internal format to STIX 2.1.
        
        Args:
            indicator: Internal threat indicator
            
        Returns:
            STIX 2.1 indicator object
        """
        try:
            # Map indicator type to STIX pattern
            pattern = self._create_stix_pattern(indicator)
            
            # Create STIX indicator object
            stix_indicator = {
                "type": "indicator",
                "spec_version": self.stix_version,
                "id": f"indicator--{indicator.indicator_id}",
                "created": indicator.first_seen.isoformat() + "Z",
                "modified": indicator.last_seen.isoformat() + "Z",
                "name": f"{indicator.threat_type} - {indicator.value}",
                "description": f"Threat indicator from {indicator.source}",
                "indicator_types": [self._map_threat_type(indicator.threat_type)],
                "pattern": pattern,
                "pattern_type": "stix",
                "valid_from": indicator.first_seen.isoformat() + "Z",
                "valid_until": indicator.last_seen.isoformat() + "Z",
                "confidence": int(indicator.confidence_score * 100),
                "labels": indicator.tags,
                "external_references": [
                    {
                        "source_name": indicator.source,
                        "description": f"Original source: {indicator.source}"
                    }
                ],
                "object_marking_refs": ["marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da"],  # TLP:GREEN
            }
            
            # Add custom properties
            if indicator.context:
                stix_indicator["x_aegis_context"] = indicator.context
            
            if indicator.false_positive_reports > 0:
                stix_indicator["x_aegis_false_positive_reports"] = indicator.false_positive_reports
            
            logger.debug(f"Converted indicator to STIX: {indicator.indicator_id}")
            return stix_indicator
            
        except Exception as e:
            logger.error(f"Failed to convert to STIX: {e}")
            raise
    
    def from_stix(self, stix_obj: Dict[str, Any]) -> ThreatIndicator:
        """
        Convert STIX 2.1 to internal format.
        
        Args:
            stix_obj: STIX 2.1 indicator object
            
        Returns:
            Internal threat indicator
        """
        try:
            # Extract indicator ID
            indicator_id = stix_obj["id"].replace("indicator--", "")
            
            # Parse STIX pattern to extract value and type
            value, indicator_type = self._parse_stix_pattern(stix_obj["pattern"])
            
            # Extract threat type
            threat_type = "unknown"
            if stix_obj.get("indicator_types"):
                threat_type = stix_obj["indicator_types"][0]
            
            # Extract confidence
            confidence = stix_obj.get("confidence", 50) / 100.0
            
            # Extract timestamps
            created = datetime.fromisoformat(stix_obj["created"].replace("Z", ""))
            modified = datetime.fromisoformat(stix_obj["modified"].replace("Z", ""))
            
            # Extract source
            source = "unknown"
            if stix_obj.get("external_references"):
                source = stix_obj["external_references"][0].get("source_name", "unknown")
            
            # Extract tags
            tags = stix_obj.get("labels", [])
            
            # Extract context
            context = stix_obj.get("x_aegis_context", {})
            
            # Extract false positive reports
            false_positive_reports = stix_obj.get("x_aegis_false_positive_reports", 0)
            
            # Create indicator
            indicator = ThreatIndicator(
                indicator_id=indicator_id,
                indicator_type=indicator_type,
                value=value,
                threat_type=threat_type,
                confidence_score=confidence,
                source=source,
                first_seen=created,
                last_seen=modified,
                tags=tags,
                context=context,
                false_positive_reports=false_positive_reports,
                validation_status=ValidationStatus.VALIDATED,
            )
            
            logger.debug(f"Converted STIX to indicator: {indicator_id}")
            return indicator
            
        except Exception as e:
            logger.error(f"Failed to convert from STIX: {e}")
            raise
    
    def _create_stix_pattern(self, indicator: ThreatIndicator) -> str:
        """
        Create STIX pattern from indicator.
        
        Args:
            indicator: Internal indicator
            
        Returns:
            STIX pattern string
        """
        if indicator.indicator_type == IndicatorType.IP:
            return f"[ipv4-addr:value = '{indicator.value}']"
        
        elif indicator.indicator_type == IndicatorType.DOMAIN:
            return f"[domain-name:value = '{indicator.value}']"
        
        elif indicator.indicator_type == IndicatorType.HASH:
            # Determine hash type by length
            if len(indicator.value) == 32:
                return f"[file:hashes.MD5 = '{indicator.value}']"
            elif len(indicator.value) == 40:
                return f"[file:hashes.SHA-1 = '{indicator.value}']"
            elif len(indicator.value) == 64:
                return f"[file:hashes.SHA-256 = '{indicator.value}']"
            else:
                return f"[file:hashes.'UNKNOWN' = '{indicator.value}']"
        
        elif indicator.indicator_type == IndicatorType.URL:
            return f"[url:value = '{indicator.value}']"
        
        elif indicator.indicator_type == IndicatorType.EMAIL:
            return f"[email-addr:value = '{indicator.value}']"
        
        elif indicator.indicator_type == IndicatorType.CERTIFICATE:
            return f"[x509-certificate:serial_number = '{indicator.value}']"
        
        else:
            return f"[unknown:value = '{indicator.value}']"
    
    def _parse_stix_pattern(self, pattern: str) -> tuple:
        """
        Parse STIX pattern to extract value and type.
        
        Args:
            pattern: STIX pattern string
            
        Returns:
            Tuple of (value, indicator_type)
        """
        import re
        
        # Extract value from pattern
        value_match = re.search(r"= '([^']+)'", pattern)
        if not value_match:
            raise ValueError(f"Could not extract value from pattern: {pattern}")
        value = value_match.group(1)
        
        # Determine type from pattern
        if "ipv4-addr" in pattern or "ipv6-addr" in pattern:
            indicator_type = IndicatorType.IP
        elif "domain-name" in pattern:
            indicator_type = IndicatorType.DOMAIN
        elif "file:hashes" in pattern:
            indicator_type = IndicatorType.HASH
        elif "url" in pattern:
            indicator_type = IndicatorType.URL
        elif "email-addr" in pattern:
            indicator_type = IndicatorType.EMAIL
        elif "x509-certificate" in pattern:
            indicator_type = IndicatorType.CERTIFICATE
        else:
            # Default to domain if unknown
            indicator_type = IndicatorType.DOMAIN
        
        return value, indicator_type
    
    def _map_threat_type(self, threat_type: str) -> str:
        """
        Map internal threat type to STIX indicator type.
        
        Args:
            threat_type: Internal threat type
            
        Returns:
            STIX indicator type
        """
        mapping = {
            "malware": "malicious-activity",
            "phishing": "phishing",
            "c2": "command-and-control",
            "botnet": "botnet",
            "ransomware": "malicious-activity",
            "exfiltration": "exfiltration",
            "dga": "domain-generation-algorithm",
        }
        
        return mapping.get(threat_type.lower(), "anomalous-activity")
    
    def validate_stix(self, stix_obj: Dict[str, Any]) -> bool:
        """
        Validate STIX object format.
        
        Args:
            stix_obj: STIX object to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["type", "spec_version", "id", "created", "modified", "pattern"]
            for field in required_fields:
                if field not in stix_obj:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check type
            if stix_obj["type"] != "indicator":
                logger.warning(f"Invalid type: {stix_obj['type']}")
                return False
            
            # Check spec version
            if not stix_obj["spec_version"].startswith("2."):
                logger.warning(f"Unsupported STIX version: {stix_obj['spec_version']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"STIX validation error: {e}")
            return False
