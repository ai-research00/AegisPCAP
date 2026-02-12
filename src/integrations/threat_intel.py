"""
Threat Intelligence Integration Module
Connects to VirusTotal, AlienVault OTX, AbuseIPDB for IP/domain reputation
"""

import logging
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json

import aiohttp
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class TILookupResult:
    """Result of a threat intelligence lookup"""
    ip_or_domain: str
    source: str  # "virustotal", "alienvault", "abuseipdb"
    threat_level: str  # "benign", "suspicious", "malicious"
    confidence: float  # 0-1
    details: Dict
    timestamp: datetime
    cached: bool = False


class VirusTotalClient:
    """VirusTotal IP/domain reputation lookups"""
    
    def __init__(self, api_key: str, cache_hours: int = 24):
        """
        Initialize VirusTotal client
        
        Args:
            api_key: VirusTotal API key
            cache_hours: How long to cache results (default 24h)
        """
        self.api_key = api_key
        self.base_url = "https://www.virustotal.com/api/v3"
        self.session = None
        self.cache_hours = cache_hours
        self._cache = {}
        self._cache_times = {}
    
    async def lookup_ip(self, ip: str) -> TILookupResult:
        """
        Lookup IP reputation on VirusTotal
        
        Args:
            ip: IP address to lookup
            
        Returns:
            TILookupResult with threat assessment
        """
        # Check cache first
        cached = self._get_cached(ip)
        if cached:
            return TILookupResult(
                ip_or_domain=ip,
                source="virustotal",
                threat_level=cached["threat_level"],
                confidence=cached["confidence"],
                details=cached["details"],
                timestamp=cached["timestamp"],
                cached=True
            )
        
        try:
            headers = {"x-apikey": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/ip_addresses/{ip}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = self._parse_vt_response(ip, data)
                        self._cache_result(ip, result)
                        return result
                    elif resp.status == 401:
                        logger.error("VirusTotal: Invalid API key")
                        raise ValueError("Invalid VirusTotal API key")
                    elif resp.status == 404:
                        logger.info(f"VirusTotal: IP {ip} not found in database")
                        return TILookupResult(
                            ip_or_domain=ip,
                            source="virustotal",
                            threat_level="benign",
                            confidence=0.3,
                            details={"message": "Not found in VirusTotal database"},
                            timestamp=datetime.now()
                        )
                    else:
                        logger.error(f"VirusTotal API error: {resp.status}")
                        raise Exception(f"VirusTotal API error: {resp.status}")
        except asyncio.TimeoutError:
            logger.error("VirusTotal: Request timeout")
            raise
        except Exception as e:
            logger.error(f"VirusTotal lookup failed: {e}")
            raise
    
    async def lookup_domain(self, domain: str) -> TILookupResult:
        """
        Lookup domain reputation on VirusTotal
        
        Args:
            domain: Domain to lookup
            
        Returns:
            TILookupResult with threat assessment
        """
        # Check cache first
        cached = self._get_cached(domain)
        if cached:
            return TILookupResult(
                ip_or_domain=domain,
                source="virustotal",
                threat_level=cached["threat_level"],
                confidence=cached["confidence"],
                details=cached["details"],
                timestamp=cached["timestamp"],
                cached=True
            )
        
        try:
            headers = {"x-apikey": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/domains/{domain}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = self._parse_vt_domain_response(domain, data)
                        self._cache_result(domain, result)
                        return result
                    else:
                        logger.warning(f"VirusTotal domain lookup failed: {resp.status}")
                        return TILookupResult(
                            ip_or_domain=domain,
                            source="virustotal",
                            threat_level="benign",
                            confidence=0.3,
                            details={"message": "Not found or error"},
                            timestamp=datetime.now()
                        )
        except Exception as e:
            logger.error(f"VirusTotal domain lookup failed: {e}")
            raise
    
    def _parse_vt_response(self, ip: str, data: Dict) -> TILookupResult:
        """Parse VirusTotal IP response"""
        attributes = data.get("data", {}).get("attributes", {})
        
        # Extract malicious detections
        last_analysis = attributes.get("last_analysis_stats", {})
        malicious_count = last_analysis.get("malicious", 0)
        suspicious_count = last_analysis.get("suspicious", 0)
        undetected_count = last_analysis.get("undetected", 0)
        total_count = sum(last_analysis.values())
        
        # Determine threat level
        if malicious_count >= 3:
            threat_level = "malicious"
            confidence = min(1.0, malicious_count / total_count) if total_count > 0 else 0.8
        elif malicious_count > 0 or suspicious_count >= 2:
            threat_level = "suspicious"
            confidence = 0.6
        else:
            threat_level = "benign"
            confidence = 0.9
        
        return TILookupResult(
            ip_or_domain=ip,
            source="virustotal",
            threat_level=threat_level,
            confidence=confidence,
            details={
                "malicious": malicious_count,
                "suspicious": suspicious_count,
                "undetected": undetected_count,
                "total_votes": total_count,
                "last_analysis_date": attributes.get("last_analysis_date"),
                "country": attributes.get("country"),
                "asn": attributes.get("asn"),
            },
            timestamp=datetime.now()
        )
    
    def _parse_vt_domain_response(self, domain: str, data: Dict) -> TILookupResult:
        """Parse VirusTotal domain response"""
        attributes = data.get("data", {}).get("attributes", {})
        
        # Extract malicious detections
        last_analysis = attributes.get("last_analysis_stats", {})
        malicious_count = last_analysis.get("malicious", 0)
        suspicious_count = last_analysis.get("suspicious", 0)
        total_count = sum(last_analysis.values())
        
        # Determine threat level
        if malicious_count >= 3:
            threat_level = "malicious"
            confidence = min(1.0, malicious_count / total_count) if total_count > 0 else 0.8
        elif malicious_count > 0 or suspicious_count >= 2:
            threat_level = "suspicious"
            confidence = 0.6
        else:
            threat_level = "benign"
            confidence = 0.9
        
        return TILookupResult(
            ip_or_domain=domain,
            source="virustotal",
            threat_level=threat_level,
            confidence=confidence,
            details={
                "malicious": malicious_count,
                "suspicious": suspicious_count,
                "total_votes": total_count,
                "categories": attributes.get("categories", {}),
                "creation_date": attributes.get("creation_date"),
                "registrar": attributes.get("registrar"),
            },
            timestamp=datetime.now()
        )
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if still valid"""
        if key in self._cache:
            cached_time = self._cache_times.get(key)
            if cached_time and datetime.now() - cached_time < timedelta(hours=self.cache_hours):
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_times[key]
        return None
    
    def _cache_result(self, key: str, result: TILookupResult):
        """Cache a result"""
        self._cache[key] = {
            "threat_level": result.threat_level,
            "confidence": result.confidence,
            "details": result.details,
            "timestamp": result.timestamp,
        }
        self._cache_times[key] = datetime.now()


class AlienVaultClient:
    """AlienVault OTX IP/domain reputation lookups"""
    
    def __init__(self, api_key: str, cache_hours: int = 24):
        """
        Initialize AlienVault OTX client
        
        Args:
            api_key: AlienVault OTX API key
            cache_hours: How long to cache results (default 24h)
        """
        self.api_key = api_key
        self.base_url = "https://otx.alienvault.com/api/v1"
        self.cache_hours = cache_hours
        self._cache = {}
        self._cache_times = {}
    
    def lookup_ip(self, ip: str) -> TILookupResult:
        """
        Lookup IP reputation on AlienVault
        
        Args:
            ip: IP address to lookup
            
        Returns:
            TILookupResult with threat assessment
        """
        # Check cache first
        cached = self._get_cached(ip)
        if cached:
            return TILookupResult(
                ip_or_domain=ip,
                source="alienvault",
                threat_level=cached["threat_level"],
                confidence=cached["confidence"],
                details=cached["details"],
                timestamp=cached["timestamp"],
                cached=True
            )
        
        try:
            headers = {"X-OTX-API-KEY": self.api_key}
            response = requests.get(
                f"{self.base_url}/indicators/ipv4/{ip}/reputation",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = self._parse_av_response(ip, data)
                self._cache_result(ip, result)
                return result
            elif response.status_code == 401:
                logger.error("AlienVault: Invalid API key")
                raise ValueError("Invalid AlienVault API key")
            elif response.status_code == 404:
                # IP not found in AlienVault database
                return TILookupResult(
                    ip_or_domain=ip,
                    source="alienvault",
                    threat_level="benign",
                    confidence=0.3,
                    details={"message": "Not found in AlienVault OTX"},
                    timestamp=datetime.now()
                )
            else:
                logger.error(f"AlienVault API error: {response.status_code}")
                raise Exception(f"AlienVault API error: {response.status_code}")
        except requests.Timeout:
            logger.error("AlienVault: Request timeout")
            raise
        except Exception as e:
            logger.error(f"AlienVault lookup failed: {e}")
            raise
    
    def lookup_domain(self, domain: str) -> TILookupResult:
        """
        Lookup domain reputation on AlienVault
        
        Args:
            domain: Domain to lookup
            
        Returns:
            TILookupResult with threat assessment
        """
        # Check cache first
        cached = self._get_cached(domain)
        if cached:
            return TILookupResult(
                ip_or_domain=domain,
                source="alienvault",
                threat_level=cached["threat_level"],
                confidence=cached["confidence"],
                details=cached["details"],
                timestamp=cached["timestamp"],
                cached=True
            )
        
        try:
            headers = {"X-OTX-API-KEY": self.api_key}
            response = requests.get(
                f"{self.base_url}/indicators/domain/{domain}/reputation",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = self._parse_av_domain_response(domain, data)
                self._cache_result(domain, result)
                return result
            else:
                logger.warning(f"AlienVault domain lookup failed: {response.status_code}")
                return TILookupResult(
                    ip_or_domain=domain,
                    source="alienvault",
                    threat_level="benign",
                    confidence=0.3,
                    details={"message": "Not found or error"},
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"AlienVault domain lookup failed: {e}")
            raise
    
    def _parse_av_response(self, ip: str, data: Dict) -> TILookupResult:
        """Parse AlienVault IP response"""
        reputation = data.get("reputation", 0)
        activities = data.get("activities", [])
        
        # Determine threat level based on reputation score
        if reputation >= 50:
            threat_level = "malicious"
            confidence = min(1.0, reputation / 100)
        elif reputation >= 20:
            threat_level = "suspicious"
            confidence = reputation / 100
        else:
            threat_level = "benign"
            confidence = max(0.3, 1.0 - reputation / 100)
        
        return TILookupResult(
            ip_or_domain=ip,
            source="alienvault",
            threat_level=threat_level,
            confidence=confidence,
            details={
                "reputation": reputation,
                "num_activities": len(activities),
                "activity_types": list(set([a.get("type") for a in activities]))[:5],
                "last_seen": activities[0].get("created") if activities else None,
            },
            timestamp=datetime.now()
        )
    
    def _parse_av_domain_response(self, domain: str, data: Dict) -> TILookupResult:
        """Parse AlienVault domain response"""
        reputation = data.get("reputation", 0)
        activities = data.get("activities", [])
        
        # Determine threat level
        if reputation >= 50:
            threat_level = "malicious"
            confidence = min(1.0, reputation / 100)
        elif reputation >= 20:
            threat_level = "suspicious"
            confidence = reputation / 100
        else:
            threat_level = "benign"
            confidence = max(0.3, 1.0 - reputation / 100)
        
        return TILookupResult(
            ip_or_domain=domain,
            source="alienvault",
            threat_level=threat_level,
            confidence=confidence,
            details={
                "reputation": reputation,
                "num_activities": len(activities),
                "activity_types": list(set([a.get("type") for a in activities]))[:5],
            },
            timestamp=datetime.now()
        )
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if still valid"""
        if key in self._cache:
            cached_time = self._cache_times.get(key)
            if cached_time and datetime.now() - cached_time < timedelta(hours=self.cache_hours):
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_times[key]
        return None
    
    def _cache_result(self, key: str, result: TILookupResult):
        """Cache a result"""
        self._cache[key] = {
            "threat_level": result.threat_level,
            "confidence": result.confidence,
            "details": result.details,
            "timestamp": result.timestamp,
        }
        self._cache_times[key] = datetime.now()


class ThreatIntelAggregator:
    """Aggregate results from multiple TI sources"""
    
    def __init__(self, vt_client: Optional[VirusTotalClient] = None, 
                 av_client: Optional[AlienVaultClient] = None):
        """
        Initialize aggregator with optional clients
        
        Args:
            vt_client: VirusTotal client (optional)
            av_client: AlienVault client (optional)
        """
        self.vt_client = vt_client
        self.av_client = av_client
    
    async def lookup_ip(self, ip: str) -> Dict:
        """
        Lookup IP across all available TI sources
        
        Args:
            ip: IP address to lookup
            
        Returns:
            Aggregated results from all sources
        """
        results = {
            "ip": ip,
            "timestamp": datetime.now().isoformat(),
            "sources": {}
        }
        
        # Try VirusTotal
        if self.vt_client:
            try:
                vt_result = await self.vt_client.lookup_ip(ip)
                results["sources"]["virustotal"] = {
                    "threat_level": vt_result.threat_level,
                    "confidence": vt_result.confidence,
                    "details": vt_result.details,
                    "cached": vt_result.cached
                }
            except Exception as e:
                logger.error(f"VirusTotal lookup failed: {e}")
                results["sources"]["virustotal"] = {"error": str(e)}
        
        # Try AlienVault
        if self.av_client:
            try:
                av_result = self.av_client.lookup_ip(ip)
                results["sources"]["alienvault"] = {
                    "threat_level": av_result.threat_level,
                    "confidence": av_result.confidence,
                    "details": av_result.details,
                    "cached": av_result.cached
                }
            except Exception as e:
                logger.error(f"AlienVault lookup failed: {e}")
                results["sources"]["alienvault"] = {"error": str(e)}
        
        # Aggregate threat level
        threat_levels = []
        confidences = []
        for source_data in results["sources"].values():
            if "threat_level" in source_data:
                threat_levels.append(source_data["threat_level"])
                confidences.append(source_data["confidence"])
        
        if threat_levels:
            # Determine overall threat level based on consensus
            malicious_count = threat_levels.count("malicious")
            suspicious_count = threat_levels.count("suspicious")
            
            if malicious_count > 0:
                results["overall_threat_level"] = "malicious"
                results["overall_confidence"] = max(confidences)
            elif suspicious_count >= len(threat_levels) * 0.5:
                results["overall_threat_level"] = "suspicious"
                results["overall_confidence"] = sum(confidences) / len(confidences)
            else:
                results["overall_threat_level"] = "benign"
                results["overall_confidence"] = sum(confidences) / len(confidences)
        else:
            results["overall_threat_level"] = "unknown"
            results["overall_confidence"] = 0.0
        
        return results
    
    async def lookup_domain(self, domain: str) -> Dict:
        """
        Lookup domain across all available TI sources
        
        Args:
            domain: Domain to lookup
            
        Returns:
            Aggregated results from all sources
        """
        results = {
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "sources": {}
        }
        
        # Try VirusTotal
        if self.vt_client:
            try:
                vt_result = await self.vt_client.lookup_domain(domain)
                results["sources"]["virustotal"] = {
                    "threat_level": vt_result.threat_level,
                    "confidence": vt_result.confidence,
                    "details": vt_result.details,
                    "cached": vt_result.cached
                }
            except Exception as e:
                logger.error(f"VirusTotal domain lookup failed: {e}")
                results["sources"]["virustotal"] = {"error": str(e)}
        
        # Try AlienVault
        if self.av_client:
            try:
                av_result = self.av_client.lookup_domain(domain)
                results["sources"]["alienvault"] = {
                    "threat_level": av_result.threat_level,
                    "confidence": av_result.confidence,
                    "details": av_result.details,
                    "cached": av_result.cached
                }
            except Exception as e:
                logger.error(f"AlienVault domain lookup failed: {e}")
                results["sources"]["alienvault"] = {"error": str(e)}
        
        # Aggregate threat level
        threat_levels = []
        confidences = []
        for source_data in results["sources"].values():
            if "threat_level" in source_data:
                threat_levels.append(source_data["threat_level"])
                confidences.append(source_data["confidence"])
        
        if threat_levels:
            malicious_count = threat_levels.count("malicious")
            suspicious_count = threat_levels.count("suspicious")
            
            if malicious_count > 0:
                results["overall_threat_level"] = "malicious"
                results["overall_confidence"] = max(confidences)
            elif suspicious_count >= len(threat_levels) * 0.5:
                results["overall_threat_level"] = "suspicious"
                results["overall_confidence"] = sum(confidences) / len(confidences)
            else:
                results["overall_threat_level"] = "benign"
                results["overall_confidence"] = sum(confidences) / len(confidences)
        else:
            results["overall_threat_level"] = "unknown"
            results["overall_confidence"] = 0.0
        
        return results
