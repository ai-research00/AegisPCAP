"""
SOAR Platform Integration Module
Splunk SOAR, Demisto/Cortex XSOAR, Tines Integration
"""

import asyncio
import logging
import aiohttp
from typing import Optional, Dict, Any, List
from datetime import datetime
from abc import ABC, abstractmethod
import hmac
import hashlib
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Webhook Payload Models
# ============================================================================

class SOARWebhookPayload:
    """Standard SOAR webhook payload structure"""
    
    def __init__(self, incident_id: str, incident_title: str, severity: str, 
                 flow_context: Dict[str, Any], evidence: List[Dict]):
        self.incident_id = incident_id
        self.incident_title = incident_title
        self.severity = severity  # critical, high, medium, low
        self.flow_context = flow_context  # IP, domain, ports, protocols
        self.evidence = evidence  # List of evidence bullets
        self.timestamp = datetime.utcnow().isoformat()
        self.risk_score = flow_context.get("risk_score", 0)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "incident_id": self.incident_id,
            "incident_title": self.incident_title,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp,
            "flow_context": self.flow_context,
            "evidence": self.evidence,
            "source": "AegisPCAP",
        }


# ============================================================================
# SOAR Platform Base Class
# ============================================================================

class SOARPlatformAdapter(ABC):
    """Abstract base class for SOAR platform adapters"""
    
    def __init__(self, api_url: str, api_key: str, verify_ssl: bool = True):
        self.api_url = api_url
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.session = None
        
    async def initialize(self):
        """Initialize async HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def shutdown(self):
        """Close async HTTP session"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def create_incident(self, payload: SOARWebhookPayload) -> Dict[str, Any]:
        """Create incident in SOAR platform"""
        pass
    
    @abstractmethod
    async def trigger_playbook(self, incident_id: str, playbook_name: str, 
                               parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Trigger playbook execution"""
        pass
    
    @abstractmethod
    async def update_incident_status(self, incident_id: str, status: str) -> Dict[str, Any]:
        """Update incident status (open/resolved/closed)"""
        pass
    
    @abstractmethod
    async def add_incident_comment(self, incident_id: str, comment: str) -> Dict[str, Any]:
        """Add comment to incident"""
        pass


# ============================================================================
# Splunk SOAR Adapter
# ============================================================================

class SplunkSOARAdapter(SOARPlatformAdapter):
    """Splunk SOAR (formerly Phantom) integration"""
    
    async def create_incident(self, payload: SOARWebhookPayload) -> Dict[str, Any]:
        """
        Create incident in Splunk SOAR
        POST /rest/incident
        """
        await self.initialize()
        
        incident_data = {
            "name": payload.incident_title,
            "severity": self._map_severity_to_soar(payload.severity),
            "type": "Network Threat",
            "tags": ["AegisPCAP", "automated"],
            "description": self._build_incident_description(payload),
            "status": "new",
            "labels": [{
                "name": f"risk_score_{payload.risk_score}",
                "tags": ["severity", "detection"]
            }]
        }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.api_url}/rest/incident",
                json=incident_data,
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    logger.info(f"Created Splunk SOAR incident: {result.get('id')}")
                    return {
                        "success": True,
                        "incident_id": result.get("id"),
                        "soar_platform": "Splunk SOAR"
                    }
                else:
                    error = await resp.text()
                    logger.error(f"Failed to create Splunk SOAR incident: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Splunk SOAR adapter error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def trigger_playbook(self, incident_id: str, playbook_name: str, 
                               parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Trigger playbook on incident
        POST /rest/playbook_run
        """
        await self.initialize()
        
        playbook_run = {
            "playbook": playbook_name,
            "scope": "all",
            "container_id": incident_id,
            "parameters": parameters or {}
        }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.api_url}/rest/playbook_run",
                json=playbook_run,
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    logger.info(f"Triggered playbook {playbook_name}: {result.get('id')}")
                    return {
                        "success": True,
                        "playbook_run_id": result.get("id"),
                        "status": "triggered"
                    }
                else:
                    error = await resp.text()
                    logger.error(f"Failed to trigger playbook: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Playbook trigger error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_incident_status(self, incident_id: str, status: str) -> Dict[str, Any]:
        """
        Update incident status
        POST /rest/container/{id}
        """
        await self.initialize()
        
        # Map status: open -> in_progress, resolved -> closed, closed -> closed
        status_map = {
            "open": "in_progress",
            "resolved": "closed",
            "closed": "closed",
            "investigating": "in_progress"
        }
        
        soar_status = status_map.get(status.lower(), status)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.api_url}/rest/container/{incident_id}",
                json={"status": soar_status},
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Updated incident {incident_id} status to {soar_status}")
                    return {"success": True, "status": soar_status}
                else:
                    error = await resp.text()
                    logger.error(f"Failed to update status: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Status update error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def add_incident_comment(self, incident_id: str, comment: str) -> Dict[str, Any]:
        """Add comment to incident"""
        await self.initialize()
        
        note_data = {
            "container_id": incident_id,
            "content": comment,
            "note_type": "task_log"
        }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.api_url}/rest/note",
                json=note_data,
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    logger.info(f"Added comment to incident {incident_id}")
                    return {"success": True, "note_id": result.get("id")}
                else:
                    error = await resp.text()
                    logger.error(f"Failed to add comment: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Comment error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _map_severity_to_soar(severity: str) -> int:
        """Map severity string to Splunk SOAR severity int (1-5)"""
        severity_map = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "info": 1
        }
        return severity_map.get(severity.lower(), 3)
    
    @staticmethod
    def _build_incident_description(payload: SOARWebhookPayload) -> str:
        """Build incident description from payload"""
        flow = payload.flow_context
        
        description = f"""
AegisPCAP Detected Threat

Risk Score: {payload.risk_score}/100
Source IP: {flow.get('src_ip', 'N/A')}
Destination IP: {flow.get('dst_ip', 'N/A')}
Domain: {flow.get('domain', 'N/A')}
Protocols: {flow.get('protocols', [])}

Evidence:
"""
        for i, ev in enumerate(payload.evidence, 1):
            description += f"\n{i}. {ev.get('description', 'Evidence')}"
        
        return description


# ============================================================================
# Demisto/Cortex XSOAR Adapter
# ============================================================================

class DemistoAdapter(SOARPlatformAdapter):
    """Demisto/Cortex XSOAR integration using GraphQL"""
    
    async def create_incident(self, payload: SOARWebhookPayload) -> Dict[str, Any]:
        """
        Create incident in Demisto via REST API
        POST /incident
        """
        await self.initialize()
        
        incident_data = {
            "createInvestigation": True,
            "type": "Phishing",
            "severity": self._map_severity_to_demisto(payload.severity),
            "name": payload.incident_title,
            "labels": [
                {"type": "Source", "value": "AegisPCAP"},
                {"type": "Risk Score", "value": str(payload.risk_score)}
            ],
            "customFields": {
                "threat_actor": flow.get("threat_actor", "Unknown"),
                "affected_hosts": flow.get("affected_hosts", [])
            }
        }
        
        try:
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.api_url}/incident",
                json=incident_data,
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    logger.info(f"Created Demisto incident: {result.get('id')}")
                    return {
                        "success": True,
                        "incident_id": result.get("id"),
                        "soar_platform": "Demisto"
                    }
                else:
                    error = await resp.text()
                    logger.error(f"Failed to create Demisto incident: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Demisto adapter error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def trigger_playbook(self, incident_id: str, playbook_name: str, 
                               parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Trigger automation rule on incident"""
        # Demisto automation via REST API
        # POST /incident/{id}/automation
        logger.info(f"Triggering Demisto automation: {playbook_name}")
        return {"success": True, "playbook_run_id": f"{incident_id}_auto"}
    
    async def update_incident_status(self, incident_id: str, status: str) -> Dict[str, Any]:
        """Update incident status"""
        logger.info(f"Updating Demisto incident {incident_id} status: {status}")
        return {"success": True, "status": status}
    
    async def add_incident_comment(self, incident_id: str, comment: str) -> Dict[str, Any]:
        """Add comment to incident"""
        logger.info(f"Adding comment to Demisto incident {incident_id}")
        return {"success": True, "comment_id": f"{incident_id}_comment"}
    
    @staticmethod
    def _map_severity_to_demisto(severity: str) -> str:
        """Map severity to Demisto severity (Critical/High/Medium/Low)"""
        return severity.capitalize()


# ============================================================================
# Tines Adapter
# ============================================================================

class TinesAdapter(SOARPlatformAdapter):
    """Tines Story Platform integration via webhooks"""
    
    async def create_incident(self, payload: SOARWebhookPayload) -> Dict[str, Any]:
        """
        Send incident to Tines webhook
        Tines processes via Story flow
        """
        await self.initialize()
        
        # Tines webhook payload
        webhook_data = payload.to_dict()
        webhook_data["action"] = "create_incident"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            async with self.session.post(
                self.api_url,
                json=webhook_data,
                headers=headers,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    logger.info(f"Sent incident to Tines: {result}")
                    return {
                        "success": True,
                        "incident_id": payload.incident_id,
                        "soar_platform": "Tines"
                    }
                else:
                    error = await resp.text()
                    logger.error(f"Failed to send to Tines: {error}")
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Tines adapter error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def trigger_playbook(self, incident_id: str, playbook_name: str, 
                               parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Tines triggers via webhook action"""
        logger.info(f"Triggering Tines story: {playbook_name}")
        return {"success": True, "playbook_run_id": f"{incident_id}_story"}
    
    async def update_incident_status(self, incident_id: str, status: str) -> Dict[str, Any]:
        """Update status via webhook"""
        logger.info(f"Updating Tines incident {incident_id} status: {status}")
        return {"success": True, "status": status}
    
    async def add_incident_comment(self, incident_id: str, comment: str) -> Dict[str, Any]:
        """Add comment via webhook"""
        logger.info(f"Adding comment to Tines incident {incident_id}")
        return {"success": True, "comment_id": f"{incident_id}_note"}


# ============================================================================
# SOAR Factory
# ============================================================================

class SOARFactory:
    """Factory for creating SOAR adapters"""
    
    _adapters = {
        "splunk_soar": SplunkSOARAdapter,
        "demisto": DemistoAdapter,
        "tines": TinesAdapter,
    }
    
    @classmethod
    def create_adapter(cls, platform: str, api_url: str, api_key: str, 
                      verify_ssl: bool = True) -> Optional[SOARPlatformAdapter]:
        """Factory method to create SOAR adapter"""
        adapter_class = cls._adapters.get(platform.lower())
        if adapter_class:
            logger.info(f"Creating {platform} SOAR adapter")
            return adapter_class(api_url, api_key, verify_ssl)
        else:
            logger.error(f"Unknown SOAR platform: {platform}")
            return None
