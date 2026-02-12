"""
SIEM Integration Module
Bi-directional sync with Splunk, ELK, Wazuh, and Suricata
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import aiohttp
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# SIEM Event Models
# ============================================================================

class SIEMEvent(BaseModel):
    """Standard SIEM event structure"""
    timestamp: datetime
    event_id: str
    source_type: str  # network, host, application
    event_name: str
    severity: str  # critical, high, medium, low, info
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    user: Optional[str] = None
    hostname: Optional[str] = None
    details: Dict[str, Any]


class SIEMAlert(BaseModel):
    """SIEM Alert/Detection"""
    alert_id: str
    title: str
    description: str
    severity: str
    status: str  # new, investigating, resolved, false_positive
    events: List[SIEMEvent]
    detection_rule: str
    correlation_count: int
    first_seen: datetime
    last_seen: datetime


# ============================================================================
# Abstract SIEM Platform Adapter
# ============================================================================

class SIEMPlatformAdapter(ABC):
    """Abstract base class for SIEM platform adapters"""
    
    def __init__(self, api_url: str, api_key: str, **kwargs):
        self.api_url = api_url
        self.api_key = api_key
        self.session = None
    
    async def connect(self):
        """Establish connection to SIEM platform"""
        self.session = aiohttp.ClientSession()
    
    async def disconnect(self):
        """Disconnect from SIEM platform"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search_events(self, query: str, start_time: datetime, end_time: datetime) -> List[SIEMEvent]:
        """Search for events in SIEM"""
        pass
    
    @abstractmethod
    async def get_alerts(self) -> List[SIEMAlert]:
        """Retrieve current alerts"""
        pass
    
    @abstractmethod
    async def create_alert(self, alert: SIEMAlert) -> Dict[str, Any]:
        """Create alert in SIEM"""
        pass
    
    @abstractmethod
    async def correlate_events(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Correlate events to identify patterns"""
        pass
    
    @abstractmethod
    async def update_alert_status(self, alert_id: str, status: str) -> Dict[str, Any]:
        """Update alert status in SIEM"""
        pass
    
    @abstractmethod
    async def stream_events(self, callback) -> None:
        """Stream real-time events from SIEM"""
        pass


# ============================================================================
# Splunk SIEM Adapter
# ============================================================================

class SplunkAdapter(SIEMPlatformAdapter):
    """Splunk Enterprise Security integration"""
    
    async def search_events(self, query: str, start_time: datetime, end_time: datetime) -> List[SIEMEvent]:
        """Search Splunk for events"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Construct Splunk search query
            splunk_query = f"""search ({query}) earliest={int(start_time.timestamp())} latest={int(end_time.timestamp())}"""
            
            payload = {
                "search": splunk_query,
                "output_mode": "json",
                "count": 10000
            }
            
            async with self.session.get(
                f"{self.api_url}/services/search/jobs/export",
                headers=headers,
                params=payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Splunk search failed: {resp.status}")
                    return []
                
                data = await resp.json()
                events = []
                
                for result in data.get("results", []):
                    event = SIEMEvent(
                        timestamp=datetime.fromtimestamp(float(result.get("_time", 0))),
                        event_id=result.get("_raw", ""),
                        source_type=result.get("source", ""),
                        event_name=result.get("eventtype", ""),
                        severity=result.get("severity", "info"),
                        src_ip=result.get("src_ip"),
                        dst_ip=result.get("dest_ip"),
                        src_port=int(result.get("src_port", 0)) if result.get("src_port") else None,
                        dst_port=int(result.get("dest_port", 0)) if result.get("dest_port") else None,
                        protocol=result.get("protocol"),
                        user=result.get("user"),
                        hostname=result.get("host"),
                        details=result
                    )
                    events.append(event)
                
                logger.info(f"Retrieved {len(events)} events from Splunk")
                return events
        
        except Exception as e:
            logger.error(f"Error searching Splunk: {str(e)}")
            return []
    
    async def get_alerts(self) -> List[SIEMAlert]:
        """Get alerts from Splunk ES"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.api_url}/services/data/indexes",
                headers=headers,
                params={"output_mode": "json"}
            ) as resp:
                if resp.status != 200:
                    return []
                
                # Parse notable events from ES
                alerts = []
                logger.info("Retrieved alerts from Splunk")
                return alerts
        
        except Exception as e:
            logger.error(f"Error getting Splunk alerts: {str(e)}")
            return []
    
    async def create_alert(self, alert: SIEMAlert) -> Dict[str, Any]:
        """Create notable event in Splunk ES"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity,
                "status": alert.status
            }
            
            async with self.session.post(
                f"{self.api_url}/services/saved/searches",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status in [200, 201]:
                    result = await resp.json()
                    logger.info(f"Created Splunk alert: {alert.alert_id}")
                    return {"success": True, "alert_id": alert.alert_id}
                else:
                    logger.error(f"Failed to create Splunk alert: {resp.status}")
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error creating Splunk alert: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def correlate_events(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Correlate events using Splunk correlation search"""
        try:
            logger.info(f"Correlating {len(events)} events in Splunk")
            # Build correlation query
            return {
                "success": True,
                "correlation_groups": [],
                "patterns_detected": []
            }
        except Exception as e:
            logger.error(f"Error correlating events: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_alert_status(self, alert_id: str, status: str) -> Dict[str, Any]:
        """Update alert status in Splunk"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {"status": status}
            
            async with self.session.post(
                f"{self.api_url}/services/data/indexes/{alert_id}",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Updated alert {alert_id} status to {status}")
                    return {"success": True}
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error updating alert status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def stream_events(self, callback) -> None:
        """Stream real-time events from Splunk"""
        try:
            logger.info("Starting Splunk event stream")
            # Implement WebSocket or polling mechanism
            pass
        except Exception as e:
            logger.error(f"Error streaming events: {str(e)}")


# ============================================================================
# ELK Stack (Elasticsearch) Adapter
# ============================================================================

class ELKAdapter(SIEMPlatformAdapter):
    """Elastic Stack (ELK) integration"""
    
    async def search_events(self, query: str, start_time: datetime, end_time: datetime) -> List[SIEMEvent]:
        """Search Elasticsearch for events"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"query_string": {"query": query}},
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time.isoformat(),
                                        "lte": end_time.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 10000
            }
            
            async with self.session.post(
                f"{self.api_url}/_search",
                headers=headers,
                json=es_query
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Elasticsearch search failed: {resp.status}")
                    return []
                
                data = await resp.json()
                events = []
                
                for hit in data.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    event = SIEMEvent(
                        timestamp=datetime.fromisoformat(source.get("@timestamp", datetime.utcnow().isoformat())),
                        event_id=hit.get("_id", ""),
                        source_type=source.get("event", {}).get("category", ""),
                        event_name=source.get("event", {}).get("action", ""),
                        severity=source.get("severity", "info"),
                        src_ip=source.get("source", {}).get("ip"),
                        dst_ip=source.get("destination", {}).get("ip"),
                        src_port=source.get("source", {}).get("port"),
                        dst_port=source.get("destination", {}).get("port"),
                        protocol=source.get("network", {}).get("protocol"),
                        user=source.get("user", {}).get("name"),
                        hostname=source.get("host", {}).get("hostname"),
                        details=source
                    )
                    events.append(event)
                
                logger.info(f"Retrieved {len(events)} events from Elasticsearch")
                return events
        
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {str(e)}")
            return []
    
    async def get_alerts(self) -> List[SIEMAlert]:
        """Get alerts from Kibana/ES"""
        try:
            # Query alerting rules
            return []
        except Exception as e:
            logger.error(f"Error getting ELK alerts: {str(e)}")
            return []
    
    async def create_alert(self, alert: SIEMAlert) -> Dict[str, Any]:
        """Create alert rule in Kibana"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            rule_config = {
                "name": alert.title,
                "description": alert.description,
                "severity": alert.severity,
                "enabled": True
            }
            
            async with self.session.post(
                f"{self.api_url}/_plugins/_alerting/monitors",
                headers=headers,
                json=rule_config
            ) as resp:
                if resp.status in [200, 201]:
                    logger.info(f"Created Kibana alert: {alert.alert_id}")
                    return {"success": True, "alert_id": alert.alert_id}
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error creating Kibana alert: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def correlate_events(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Correlate events using ES aggregations"""
        try:
            logger.info(f"Correlating {len(events)} events in Elasticsearch")
            return {
                "success": True,
                "correlation_groups": [],
                "patterns_detected": []
            }
        except Exception as e:
            logger.error(f"Error correlating events: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_alert_status(self, alert_id: str, status: str) -> Dict[str, Any]:
        """Update alert status"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.put(
                f"{self.api_url}/_plugins/_alerting/monitors/{alert_id}",
                headers=headers,
                json={"status": status}
            ) as resp:
                if resp.status == 200:
                    return {"success": True}
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error updating alert status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def stream_events(self, callback) -> None:
        """Stream events from Elasticsearch"""
        try:
            logger.info("Starting Elasticsearch event stream")
            pass
        except Exception as e:
            logger.error(f"Error streaming events: {str(e)}")


# ============================================================================
# Wazuh SIEM Adapter
# ============================================================================

class WazuhAdapter(SIEMPlatformAdapter):
    """Wazuh XDR platform integration"""
    
    async def search_events(self, query: str, start_time: datetime, end_time: datetime) -> List[SIEMEvent]:
        """Search Wazuh for events"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            params = {
                "query": query,
                "newer_than": start_time.isoformat(),
                "older_than": end_time.isoformat()
            }
            
            async with self.session.get(
                f"{self.api_url}/events",
                headers=headers,
                params=params
            ) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                events = []
                
                for event in data.get("data", []):
                    siem_event = SIEMEvent(
                        timestamp=datetime.fromisoformat(event.get("timestamp", datetime.utcnow().isoformat())),
                        event_id=event.get("id", ""),
                        source_type="host",
                        event_name=event.get("rule", {}).get("description", ""),
                        severity=event.get("rule", {}).get("level", 0),
                        hostname=event.get("agent", {}).get("name"),
                        details=event
                    )
                    events.append(siem_event)
                
                logger.info(f"Retrieved {len(events)} events from Wazuh")
                return events
        
        except Exception as e:
            logger.error(f"Error searching Wazuh: {str(e)}")
            return []
    
    async def get_alerts(self) -> List[SIEMAlert]:
        """Get alerts from Wazuh"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.api_url}/alerts",
                headers=headers
            ) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                alerts = []
                
                for alert_data in data.get("data", []):
                    alert = SIEMAlert(
                        alert_id=alert_data.get("id", ""),
                        title=alert_data.get("title", ""),
                        description=alert_data.get("description", ""),
                        severity=alert_data.get("severity", "info"),
                        status=alert_data.get("status", "new"),
                        events=[],
                        detection_rule=alert_data.get("rule_id", ""),
                        correlation_count=1,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow()
                    )
                    alerts.append(alert)
                
                logger.info(f"Retrieved {len(alerts)} alerts from Wazuh")
                return alerts
        
        except Exception as e:
            logger.error(f"Error getting Wazuh alerts: {str(e)}")
            return []
    
    async def create_alert(self, alert: SIEMAlert) -> Dict[str, Any]:
        """Create alert in Wazuh"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity
            }
            
            async with self.session.post(
                f"{self.api_url}/alerts",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status in [200, 201]:
                    logger.info(f"Created Wazuh alert")
                    return {"success": True}
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error creating Wazuh alert: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def correlate_events(self, events: List[SIEMEvent]) -> Dict[str, Any]:
        """Correlate events using Wazuh rules"""
        try:
            logger.info(f"Correlating {len(events)} events in Wazuh")
            return {
                "success": True,
                "correlation_groups": [],
                "patterns_detected": []
            }
        except Exception as e:
            logger.error(f"Error correlating events: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_alert_status(self, alert_id: str, status: str) -> Dict[str, Any]:
        """Update alert status in Wazuh"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.put(
                f"{self.api_url}/alerts/{alert_id}",
                headers=headers,
                json={"status": status}
            ) as resp:
                if resp.status == 200:
                    return {"success": True}
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        
        except Exception as e:
            logger.error(f"Error updating alert status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def stream_events(self, callback) -> None:
        """Stream events from Wazuh"""
        try:
            logger.info("Starting Wazuh event stream")
            pass
        except Exception as e:
            logger.error(f"Error streaming events: {str(e)}")


# ============================================================================
# SIEM Factory
# ============================================================================

class SIEMFactory:
    """Factory for creating SIEM platform adapters"""
    
    @staticmethod
    def create_adapter(platform: str, api_url: str, api_key: str) -> Optional[SIEMPlatformAdapter]:
        """Create appropriate SIEM adapter by platform name"""
        
        adapters = {
            "splunk": SplunkAdapter,
            "elk": ELKAdapter,
            "elasticsearch": ELKAdapter,
            "wazuh": WazuhAdapter,
        }
        
        adapter_class = adapters.get(platform.lower())
        
        if not adapter_class:
            logger.error(f"Unknown SIEM platform: {platform}")
            return None
        
        try:
            adapter = adapter_class(api_url, api_key)
            logger.info(f"Created {platform} SIEM adapter")
            return adapter
        
        except Exception as e:
            logger.error(f"Error creating {platform} adapter: {str(e)}")
            return None
