"""
Data Anonymizer for Research API

Removes PII from flows and alerts for community research access.
Builds on Phase 13 anonymization infrastructure.

Type hints: 100% coverage
Docstrings: 100% coverage
"""

from typing import Any, Dict, List, Optional
import hashlib
import re
import logging

from src.compliance.anonymization import AnonymizationController


# ============================================================================
# DATA ANONYMIZER
# ============================================================================

class DataAnonymizer:
    """
    Anonymize network flows and alerts for research access.
    
    Removes or obfuscates:
    - IP addresses
    - Hostnames and domains
    - MAC addresses
    - User identifiers
    - Sensitive payloads
    """
    
    def __init__(self):
        """Initialize data anonymizer."""
        self.logger = logging.getLogger(__name__)
        self.anonymization_controller = AnonymizationController()
        
        # Track anonymization mappings for consistency
        self.ip_mappings: Dict[str, str] = {}
        self.domain_mappings: Dict[str, str] = {}
        self.mac_mappings: Dict[str, str] = {}
    
    def _anonymize_ip(self, ip_address: str) -> str:
        """
        Anonymize IP address consistently.
        
        Args:
            ip_address: Original IP address
            
        Returns:
            Anonymized IP address
        """
        if not ip_address or ip_address == "0.0.0.0":
            return "0.0.0.0"
        
        # Check if already anonymized
        if ip_address in self.ip_mappings:
            return self.ip_mappings[ip_address]
        
        # Determine if private or public IP
        is_private = (
            ip_address.startswith("10.") or
            ip_address.startswith("192.168.") or
            ip_address.startswith("172.16.") or
            ip_address.startswith("172.17.") or
            ip_address.startswith("172.18.") or
            ip_address.startswith("172.19.") or
            ip_address.startswith("172.20.") or
            ip_address.startswith("172.21.") or
            ip_address.startswith("172.22.") or
            ip_address.startswith("172.23.") or
            ip_address.startswith("172.24.") or
            ip_address.startswith("172.25.") or
            ip_address.startswith("172.26.") or
            ip_address.startswith("172.27.") or
            ip_address.startswith("172.28.") or
            ip_address.startswith("172.29.") or
            ip_address.startswith("172.30.") or
            ip_address.startswith("172.31.")
        )
        
        # Create consistent hash-based anonymization
        hash_value = hashlib.sha256(ip_address.encode()).hexdigest()[:8]
        hash_int = int(hash_value, 16)
        
        if is_private:
            # Map to 10.x.x.x range
            octet2 = (hash_int >> 16) & 0xFF
            octet3 = (hash_int >> 8) & 0xFF
            octet4 = hash_int & 0xFF
            anonymized = f"10.{octet2}.{octet3}.{octet4}"
        else:
            # Map to 203.0.113.x (TEST-NET-3 reserved range)
            octet4 = hash_int & 0xFF
            anonymized = f"203.0.113.{octet4}"
        
        self.ip_mappings[ip_address] = anonymized
        return anonymized
    
    def _anonymize_domain(self, domain: str) -> str:
        """
        Anonymize domain name.
        
        Args:
            domain: Original domain
            
        Returns:
            Anonymized domain
        """
        if not domain:
            return ""
        
        # Check if already anonymized
        if domain in self.domain_mappings:
            return self.domain_mappings[domain]
        
        # Extract TLD
        parts = domain.split(".")
        if len(parts) < 2:
            return "example.com"
        
        tld = parts[-1]
        
        # Create hash-based anonymization
        hash_value = hashlib.sha256(domain.encode()).hexdigest()[:12]
        anonymized = f"domain-{hash_value}.{tld}"
        
        self.domain_mappings[domain] = anonymized
        return anonymized
    
    def _anonymize_mac(self, mac_address: str) -> str:
        """
        Anonymize MAC address.
        
        Args:
            mac_address: Original MAC address
            
        Returns:
            Anonymized MAC address
        """
        if not mac_address:
            return ""
        
        # Check if already anonymized
        if mac_address in self.mac_mappings:
            return self.mac_mappings[mac_address]
        
        # Create hash-based anonymization
        hash_value = hashlib.sha256(mac_address.encode()).hexdigest()[:12]
        # Format as MAC address
        anonymized = ":".join([hash_value[i:i+2] for i in range(0, 12, 2)])
        
        self.mac_mappings[mac_address] = anonymized
        return anonymized
    
    def _remove_sensitive_patterns(self, text: str) -> str:
        """
        Remove sensitive patterns from text.
        
        Args:
            text: Original text
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove credit card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        # Remove SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
    
    def anonymize_flows(self, flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize network flows for research access.
        
        Args:
            flows: List of flow records
            
        Returns:
            Anonymized flow records
        """
        anonymized_flows = []
        
        for flow in flows:
            anon_flow = flow.copy()
            
            # Anonymize IP addresses
            if "src_ip" in anon_flow:
                anon_flow["src_ip"] = self._anonymize_ip(anon_flow["src_ip"])
            
            if "dst_ip" in anon_flow:
                anon_flow["dst_ip"] = self._anonymize_ip(anon_flow["dst_ip"])
            
            if "source_ip" in anon_flow:
                anon_flow["source_ip"] = self._anonymize_ip(anon_flow["source_ip"])
            
            if "target_ip" in anon_flow:
                anon_flow["target_ip"] = self._anonymize_ip(anon_flow["target_ip"])
            
            # Anonymize hostnames/domains
            if "src_hostname" in anon_flow:
                anon_flow["src_hostname"] = self._anonymize_domain(anon_flow["src_hostname"])
            
            if "dst_hostname" in anon_flow:
                anon_flow["dst_hostname"] = self._anonymize_domain(anon_flow["dst_hostname"])
            
            if "dns_query" in anon_flow:
                anon_flow["dns_query"] = self._anonymize_domain(anon_flow["dns_query"])
            
            # Anonymize MAC addresses
            if "src_mac" in anon_flow:
                anon_flow["src_mac"] = self._anonymize_mac(anon_flow["src_mac"])
            
            if "dst_mac" in anon_flow:
                anon_flow["dst_mac"] = self._anonymize_mac(anon_flow["dst_mac"])
            
            # Remove user identifiers
            if "user_id" in anon_flow:
                anon_flow["user_id"] = "[REDACTED]"
            
            if "username" in anon_flow:
                anon_flow["username"] = "[REDACTED]"
            
            # Sanitize payload if present
            if "payload" in anon_flow:
                anon_flow["payload"] = self._remove_sensitive_patterns(anon_flow["payload"])
            
            # Mark as anonymized
            anon_flow["_anonymized"] = True
            
            anonymized_flows.append(anon_flow)
        
        self.logger.info(f"Anonymized {len(flows)} flows")
        return anonymized_flows
    
    def anonymize_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize threat alerts for research access.
        
        Args:
            alerts: List of alert records
            
        Returns:
            Anonymized alert records
        """
        anonymized_alerts = []
        
        for alert in alerts:
            anon_alert = alert.copy()
            
            # Anonymize IP addresses
            if "source_ip" in anon_alert:
                anon_alert["source_ip"] = self._anonymize_ip(anon_alert["source_ip"])
            
            if "target_ip" in anon_alert:
                anon_alert["target_ip"] = self._anonymize_ip(anon_alert["target_ip"])
            
            if "src_ip" in anon_alert:
                anon_alert["src_ip"] = self._anonymize_ip(anon_alert["src_ip"])
            
            if "dst_ip" in anon_alert:
                anon_alert["dst_ip"] = self._anonymize_ip(anon_alert["dst_ip"])
            
            # Anonymize hostnames
            if "hostname" in anon_alert:
                anon_alert["hostname"] = self._anonymize_domain(anon_alert["hostname"])
            
            if "domain" in anon_alert:
                anon_alert["domain"] = self._anonymize_domain(anon_alert["domain"])
            
            # Remove user identifiers
            if "user_id" in anon_alert:
                anon_alert["user_id"] = "[REDACTED]"
            
            if "affected_user" in anon_alert:
                anon_alert["affected_user"] = "[REDACTED]"
            
            # Sanitize description and details
            if "description" in anon_alert:
                anon_alert["description"] = self._remove_sensitive_patterns(anon_alert["description"])
            
            if "details" in anon_alert:
                anon_alert["details"] = self._remove_sensitive_patterns(str(anon_alert["details"]))
            
            # Keep threat intelligence fields (non-PII)
            # - threat_type, confidence, severity, mitre_tactics, etc.
            
            # Mark as anonymized
            anon_alert["_anonymized"] = True
            
            anonymized_alerts.append(anon_alert)
        
        self.logger.info(f"Anonymized {len(alerts)} alerts")
        return anonymized_alerts
    
    def anonymize_statistics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize statistical data.
        
        Args:
            stats: Statistical summary
            
        Returns:
            Anonymized statistics
        """
        # Statistics are typically already aggregated and don't contain PII
        # But we should still check for any IP addresses or domains
        
        anon_stats = {}
        
        for key, value in stats.items():
            if isinstance(value, str):
                # Check if it's an IP or domain
                if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', value):
                    anon_stats[key] = self._anonymize_ip(value)
                elif "." in value and len(value.split(".")) >= 2:
                    anon_stats[key] = self._anonymize_domain(value)
                else:
                    anon_stats[key] = value
            elif isinstance(value, dict):
                anon_stats[key] = self.anonymize_statistics(value)
            elif isinstance(value, list):
                anon_stats[key] = [
                    self.anonymize_statistics(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                anon_stats[key] = value
        
        return anon_stats
    
    def get_anonymization_report(self) -> Dict[str, Any]:
        """
        Get anonymization statistics.
        
        Returns:
            Anonymization report
        """
        return {
            "ips_anonymized": len(self.ip_mappings),
            "domains_anonymized": len(self.domain_mappings),
            "macs_anonymized": len(self.mac_mappings),
            "anonymization_method": "hash-based_consistent_mapping",
            "pii_patterns_removed": [
                "email_addresses",
                "phone_numbers",
                "credit_cards",
                "ssn",
                "user_identifiers"
            ]
        }
    
    def reset_mappings(self) -> None:
        """Reset anonymization mappings (for testing or new sessions)."""
        self.ip_mappings.clear()
        self.domain_mappings.clear()
        self.mac_mappings.clear()
        self.logger.info("Anonymization mappings reset")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DataAnonymizer"
]
