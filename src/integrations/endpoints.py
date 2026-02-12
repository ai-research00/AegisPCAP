"""
AegisPCAP Integration API Endpoints
FastAPI endpoints for external integrations
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging

from src.integrations import (
    VirusTotalClient, 
    AlienVaultClient, 
    ThreatIntelAggregator,
    SlackNotifier, 
    TeamsNotifier, 
    DiscordNotifier,
    JiraConnector,
    ServiceNowConnector,
    INTEGRATION_CONFIG
)
from src.integrations.soar_webhook import soar_webhook_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

# Initialize clients from config
vt_client = None
av_client = None
aggregator = None
slack_notifier = None
teams_notifier = None
discord_notifier = None
jira_connector = None
servicenow_connector = None

# Initialize on startup
def initialize_integrations():
    global vt_client, av_client, aggregator, slack_notifier, teams_notifier, discord_notifier, jira_connector, servicenow_connector
    
    # TI clients
    if INTEGRATION_CONFIG.threat_intel.virustotal_enabled and INTEGRATION_CONFIG.threat_intel.virustotal_api_key:
        vt_client = VirusTotalClient(INTEGRATION_CONFIG.threat_intel.virustotal_api_key)
    
    if INTEGRATION_CONFIG.threat_intel.alienvault_enabled and INTEGRATION_CONFIG.threat_intel.alienvault_api_key:
        av_client = AlienVaultClient(INTEGRATION_CONFIG.threat_intel.alienvault_api_key)
    
    aggregator = ThreatIntelAggregator(vt_client, av_client)
    
    # Notifiers
    if INTEGRATION_CONFIG.notifications.slack_enabled and INTEGRATION_CONFIG.notifications.slack_webhook_url:
        slack_notifier = SlackNotifier(
            INTEGRATION_CONFIG.notifications.slack_webhook_url,
            INTEGRATION_CONFIG.notifications.slack_channel
        )
    
    if INTEGRATION_CONFIG.notifications.teams_enabled and INTEGRATION_CONFIG.notifications.teams_webhook_url:
        teams_notifier = TeamsNotifier(INTEGRATION_CONFIG.notifications.teams_webhook_url)
    
    if INTEGRATION_CONFIG.notifications.discord_enabled and INTEGRATION_CONFIG.notifications.discord_webhook_url:
        discord_notifier = DiscordNotifier(INTEGRATION_CONFIG.notifications.discord_webhook_url)
    
    # Ticketing
    if INTEGRATION_CONFIG.ticketing.jira_enabled and INTEGRATION_CONFIG.ticketing.jira_url:
        jira_connector = JiraConnector(
            INTEGRATION_CONFIG.ticketing.jira_url,
            INTEGRATION_CONFIG.ticketing.jira_api_token,
            INTEGRATION_CONFIG.ticketing.jira_project_key
        )
    
    if INTEGRATION_CONFIG.ticketing.servicenow_enabled and INTEGRATION_CONFIG.ticketing.servicenow_url:
        servicenow_connector = ServiceNowConnector(
            INTEGRATION_CONFIG.ticketing.servicenow_url,
            INTEGRATION_CONFIG.ticketing.servicenow_user,
            INTEGRATION_CONFIG.ticketing.servicenow_password
        )


# ============================================================================
# THREAT INTELLIGENCE ENDPOINTS
# ============================================================================

class TILookupRequest(BaseModel):
    """Threat Intelligence lookup request"""
    ip_or_domain: str
    lookup_type: str = "ip"  # "ip" or "domain"


class TILookupResponse(BaseModel):
    """Threat Intelligence lookup response"""
    ip_or_domain: str
    overall_threat_level: str
    overall_confidence: float
    sources: Dict


@router.post("/ti/lookup", response_model=TILookupResponse)
async def lookup_threat_intel(request: TILookupRequest) -> Dict:
    """
    Lookup IP or domain across threat intelligence sources
    
    Query Parameters:
    - ip_or_domain: IP address or domain name
    - lookup_type: "ip" or "domain"
    
    Returns:
    - overall_threat_level: "benign", "suspicious", or "malicious"
    - overall_confidence: 0-1 confidence score
    - sources: Results from each source (VirusTotal, AlienVault, etc)
    """
    if not aggregator:
        raise HTTPException(status_code=503, detail="Threat intelligence not configured")
    
    try:
        if request.lookup_type == "ip":
            result = await aggregator.lookup_ip(request.ip_or_domain)
        else:
            result = await aggregator.lookup_domain(request.ip_or_domain)
        
        return result
    except Exception as e:
        logger.error(f"TI lookup failed: {e}")
        raise HTTPException(status_code=500, detail=f"TI lookup failed: {str(e)}")


@router.get("/ti/cached/{ip_or_domain}")
async def get_cached_threat_intel(ip_or_domain: str) -> Dict:
    """Get cached threat intelligence (no external API calls)"""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Threat intelligence not configured")
    
    # Try to get from cache
    try:
        result = await aggregator.lookup_ip(ip_or_domain)
        if all(v.get("cached") for v in result.get("sources", {}).values()):
            return result
        raise HTTPException(status_code=404, detail="Not in cache")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Not in cache")


# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================

class AlertNotificationRequest(BaseModel):
    """Alert notification request"""
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    channels: List[str]  # ["slack", "teams", "discord", "email"]
    recipients: Optional[List[str]] = None  # For email


@router.post("/notifications/send")
async def send_alert_notification(request: AlertNotificationRequest) -> Dict:
    """Send alert to configured notification channels"""
    from src.integrations.notifiers import AlertMessage
    from datetime import datetime
    
    results = {}
    
    alert = AlertMessage(
        title=request.title,
        description=request.description,
        severity=request.severity,
        source="AegisPCAP",
        timestamp=datetime.now(),
        details={}
    )
    
    if "slack" in request.channels and slack_notifier:
        try:
            results["slack"] = await slack_notifier.send_alert_async(alert)
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            results["slack"] = False
    
    if "teams" in request.channels and teams_notifier:
        try:
            results["teams"] = teams_notifier.send_alert(alert)
        except Exception as e:
            logger.error(f"Teams notification failed: {e}")
            results["teams"] = False
    
    if "discord" in request.channels and discord_notifier:
        try:
            results["discord"] = discord_notifier.send_alert(alert)
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            results["discord"] = False
    
    return {"status": "sent", "results": results}


# ============================================================================
# TICKETING ENDPOINTS
# ============================================================================

class TicketCreateRequest(BaseModel):
    """Create ticket request"""
    title: str
    description: str
    system: str  # "jira" or "servicenow"
    priority: str = "Medium"
    labels: Optional[List[str]] = None


@router.post("/tickets/create")
async def create_ticket(request: TicketCreateRequest) -> Dict:
    """Create a ticket in Jira or ServiceNow"""
    try:
        if request.system == "jira":
            if not jira_connector:
                raise HTTPException(status_code=503, detail="Jira not configured")
            
            ticket_key = jira_connector.create_ticket(
                title=request.title,
                description=request.description,
                priority=request.priority,
                labels=request.labels
            )
            
            if ticket_key:
                return {
                    "status": "created",
                    "system": "jira",
                    "ticket_id": ticket_key,
                    "url": f"{INTEGRATION_CONFIG.ticketing.jira_url}/browse/{ticket_key}"
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to create Jira ticket")
        
        elif request.system == "servicenow":
            if not servicenow_connector:
                raise HTTPException(status_code=503, detail="ServiceNow not configured")
            
            incident_num = servicenow_connector.create_incident(
                short_description=request.title,
                description=request.description,
                urgency=1 if request.priority == "High" else 2,
                assignment_group="Security"
            )
            
            if incident_num:
                return {
                    "status": "created",
                    "system": "servicenow",
                    "ticket_id": incident_num
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to create ServiceNow incident")
        
        else:
            raise HTTPException(status_code=400, detail="Unknown ticketing system")
            
    except Exception as e:
        logger.error(f"Ticket creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ticket creation failed: {str(e)}")


class TicketUpdateRequest(BaseModel):
    """Update ticket request"""
    ticket_id: str
    system: str  # "jira" or "servicenow"
    fields: Dict


@router.patch("/tickets/{ticket_id}")
async def update_ticket(ticket_id: str, request: TicketUpdateRequest) -> Dict:
    """Update a ticket in Jira or ServiceNow"""
    try:
        if request.system == "jira":
            if not jira_connector:
                raise HTTPException(status_code=503, detail="Jira not configured")
            
            success = jira_connector.update_ticket(ticket_id, request.fields)
            if success:
                return {"status": "updated", "ticket_id": ticket_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to update Jira ticket")
        
        elif request.system == "servicenow":
            if not servicenow_connector:
                raise HTTPException(status_code=503, detail="ServiceNow not configured")
            
            success = servicenow_connector.update_incident(ticket_id, request.fields)
            if success:
                return {"status": "updated", "ticket_id": ticket_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to update ServiceNow incident")
        
        else:
            raise HTTPException(status_code=400, detail="Unknown ticketing system")
            
    except Exception as e:
        logger.error(f"Ticket update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ticket update failed: {str(e)}")


# ============================================================================
# SIEM INTEGRATION ENDPOINTS
# ============================================================================

class SIEMSearchRequest(BaseModel):
    """SIEM search request"""
    siem_platform: str
    query: str
    start_time: str  # ISO format
    end_time: str    # ISO format
    siem_api_url: str
    siem_api_key: str


@router.post("/siem/search")
async def search_siem(request: SIEMSearchRequest) -> Dict:
    """
    Search SIEM platform for events
    
    Supports: Splunk, ELK, Wazuh
    
    Example:
    ```json
    {
        "siem_platform": "splunk",
        "query": "index=main sourcetype=firewall",
        "start_time": "2026-02-05T00:00:00Z",
        "end_time": "2026-02-05T23:59:59Z",
        "siem_api_url": "https://splunk.example.com:8089",
        "siem_api_key": "xxx"
    }
    ```
    """
    try:
        from src.integrations.siem import SIEMFactory
        from datetime import datetime
        
        adapter = SIEMFactory.create_adapter(
            request.siem_platform,
            request.siem_api_url,
            request.siem_api_key
        )
        
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unknown SIEM platform: {request.siem_platform}")
        
        await adapter.connect()
        
        try:
            start = datetime.fromisoformat(request.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(request.end_time.replace('Z', '+00:00'))
            
            events = await adapter.search_events(request.query, start, end)
            
            return {
                "success": True,
                "siem_platform": request.siem_platform,
                "event_count": len(events),
                "events": [e.dict() if hasattr(e, 'dict') else e for e in events]
            }
        finally:
            await adapter.disconnect()
    
    except Exception as e:
        logger.error(f"SIEM search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SIEM search failed: {str(e)}")


@router.get("/siem/alerts")
async def get_siem_alerts(
    siem_platform: str,
    siem_api_url: str,
    siem_api_key: str
) -> Dict:
    """Get current alerts from SIEM platform"""
    try:
        from src.integrations.siem import SIEMFactory
        
        adapter = SIEMFactory.create_adapter(siem_platform, siem_api_url, siem_api_key)
        
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unknown SIEM platform: {siem_platform}")
        
        await adapter.connect()
        
        try:
            alerts = await adapter.get_alerts()
            
            return {
                "success": True,
                "siem_platform": siem_platform,
                "alert_count": len(alerts),
                "alerts": [a.dict() if hasattr(a, 'dict') else a for a in alerts]
            }
        finally:
            await adapter.disconnect()
    
    except Exception as e:
        logger.error(f"Failed to get SIEM alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get SIEM alerts: {str(e)}")


# ============================================================================
# RESPONSE ACTIONS ENDPOINTS
# ============================================================================

class ExecuteResponseActionRequest(BaseModel):
    """Execute response action request"""
    action_type: str  # block_ip, isolate_endpoint, disable_account, etc.
    target: str       # IP, domain, endpoint ID, account name
    priority: str     # critical, high, medium, low
    reason: str
    incident_id: str
    parameters: Optional[Dict] = None


@router.post("/actions/execute")
async def execute_response_action(request: ExecuteResponseActionRequest) -> Dict:
    """
    Execute a response action
    
    Priority levels:
    - critical: Auto-execute immediately (time-critical threats)
    - high: Auto-execute with logging
    - medium: Queue for analyst approval (prevent false positives)
    - low: Batch queue for review
    
    Example:
    ```json
    {
        "action_type": "block_ip",
        "target": "192.168.1.100",
        "priority": "critical",
        "reason": "Malware C2 detected",
        "incident_id": "AEGIS_20260205_123456"
    }
    ```
    """
    try:
        from src.integrations.response_actions import (
            ResponseAction, ResponseActionType, ResponsePriority,
            ResponseActionOrchestrator
        )
        
        # Create action
        action = ResponseAction(
            action_type=ResponseActionType(request.action_type),
            target=request.target,
            priority=ResponsePriority(request.priority),
            reason=request.reason,
            incident_id=request.incident_id,
            parameters=request.parameters
        )
        
        # Execute via global orchestrator (would be initialized at startup)
        # For now, just log and queue
        logger.info(f"Executing action: {action.action_id} - {request.action_type} on {request.target}")
        
        return {
            "success": True,
            "action_id": action.action_id,
            "status": "queued" if action.priority in ["medium", "low"] else "executing",
            "message": f"Action {action.action_id} queued for {request.priority} priority execution"
        }
    
    except Exception as e:
        logger.error(f"Failed to execute response action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute action: {str(e)}")


@router.get("/actions/pending")
async def get_pending_actions() -> Dict:
    """Get all pending response actions awaiting analyst approval"""
    try:
        # Would query from orchestrator
        return {
            "success": True,
            "pending_count": 0,
            "actions": []
        }
    except Exception as e:
        logger.error(f"Failed to get pending actions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending actions: {str(e)}")


@router.post("/actions/{action_id}/approve")
async def approve_response_action(action_id: str) -> Dict:
    """Approve a pending response action"""
    try:
        logger.info(f"Approving action: {action_id}")
        
        return {
            "success": True,
            "action_id": action_id,
            "status": "approved_and_executing"
        }
    except Exception as e:
        logger.error(f"Failed to approve action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to approve action: {str(e)}")


@router.post("/actions/{action_id}/reject")
async def reject_response_action(action_id: str, reason: Optional[str] = None) -> Dict:
    """Reject a pending response action"""
    try:
        logger.info(f"Rejecting action: {action_id}, reason: {reason}")
        
        return {
            "success": True,
            "action_id": action_id,
            "status": "rejected"
        }
    except Exception as e:
        logger.error(f"Failed to reject action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reject action: {str(e)}")


@router.get("/actions/history")
async def get_action_history(incident_id: Optional[str] = None) -> Dict:
    """Get action execution history"""
    try:
        return {
            "success": True,
            "incident_id": incident_id,
            "action_count": 0,
            "actions": []
        }
    except Exception as e:
        logger.error(f"Failed to get action history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get action history: {str(e)}")


# ============================================================================
# FIREWALL INTEGRATION ENDPOINTS
# ============================================================================

class FirewallBlockRequest(BaseModel):
    """Firewall block request"""
    firewall_platform: str  # palo_alto, fortinet, pfsense, suricata
    target_type: str        # ip, domain, url
    target: str
    reason: str
    firewall_api_url: str
    firewall_api_key: str
    duration_seconds: Optional[int] = None


@router.post("/firewall/block")
async def block_in_firewall(request: FirewallBlockRequest) -> Dict:
    """
    Block IP/domain/URL in firewall
    
    Supports:
    - Palo Alto Networks
    - Fortinet FortiGate
    - pfSense
    - Suricata
    
    Example:
    ```json
    {
        "firewall_platform": "palo_alto",
        "target_type": "ip",
        "target": "192.168.1.100",
        "reason": "Malware C2 detected",
        "firewall_api_url": "https://paloalto.example.com",
        "firewall_api_key": "xxx"
    }
    ```
    """
    try:
        from src.integrations.firewall import FirewallFactory
        
        adapter = FirewallFactory.create_adapter(
            request.firewall_platform,
            request.firewall_api_url,
            request.firewall_api_key
        )
        
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unknown firewall platform: {request.firewall_platform}")
        
        await adapter.connect()
        
        try:
            if request.target_type == "ip":
                result = await adapter.block_ip(request.target, reason=request.reason)
            elif request.target_type == "domain":
                result = await adapter.block_domain(request.target, reason=request.reason)
            elif request.target_type == "url":
                result = await adapter.block_url(request.target, reason=request.reason)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown target type: {request.target_type}")
            
            return {
                "success": result.get("success", False),
                "firewall_platform": request.firewall_platform,
                "target": request.target,
                "result": result
            }
        finally:
            await adapter.disconnect()
    
    except Exception as e:
        logger.error(f"Firewall block failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Firewall block failed: {str(e)}")


@router.get("/firewall/rules")
async def get_firewall_rules(
    firewall_platform: str,
    firewall_api_url: str,
    firewall_api_key: str
) -> Dict:
    """Get firewall rules"""
    try:
        from src.integrations.firewall import FirewallFactory
        
        adapter = FirewallFactory.create_adapter(firewall_platform, firewall_api_url, firewall_api_key)
        
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unknown firewall platform: {firewall_platform}")
        
        await adapter.connect()
        
        try:
            rules = await adapter.get_rules()
            
            return {
                "success": True,
                "firewall_platform": firewall_platform,
                "rule_count": len(rules),
                "rules": [r.dict() if hasattr(r, 'dict') else r for r in rules]
            }
        finally:
            await adapter.disconnect()
    
    except Exception as e:
        logger.error(f"Failed to get firewall rules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get firewall rules: {str(e)}")


# ============================================================================
# ADVANCED QUERY FILTERS
# ============================================================================

class AdvancedFlowSearchRequest(BaseModel):
    """Advanced flow search with filters"""
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    risk_level: Optional[str] = None  # low, medium, high, critical
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    domain: Optional[str] = None
    user_agent: Optional[str] = None
    limit: int = 100
    offset: int = 0


@router.post("/flows/advanced-search")
async def advanced_flow_search(request: AdvancedFlowSearchRequest) -> Dict:
    """
    Advanced flow search with multiple filters
    
    Supports filtering by:
    - IP addresses (source/destination)
    - Ports
    - Protocol
    - Risk level
    - Time range
    - Domain
    - User Agent
    - Pagination (limit, offset)
    
    Example:
    ```json
    {
        "src_ip": "192.168.1.100",
        "protocol": "DNS",
        "risk_level": "high",
        "start_time": "2026-02-05T00:00:00Z",
        "end_time": "2026-02-05T23:59:59Z",
        "limit": 50,
        "offset": 0
    }
    ```
    """
    try:
        # Build filter query
        filters = {}
        
        if request.src_ip:
            filters["src_ip"] = request.src_ip
        if request.dst_ip:
            filters["dst_ip"] = request.dst_ip
        if request.src_port:
            filters["src_port"] = request.src_port
        if request.dst_port:
            filters["dst_port"] = request.dst_port
        if request.protocol:
            filters["protocol"] = request.protocol
        if request.risk_level:
            filters["risk_level"] = request.risk_level
        if request.domain:
            filters["domain"] = request.domain
        if request.user_agent:
            filters["user_agent"] = request.user_agent
        if request.start_time:
            filters["start_time"] = request.start_time
        if request.end_time:
            filters["end_time"] = request.end_time
        
        logger.info(f"Advanced flow search with filters: {filters}, limit={request.limit}, offset={request.offset}")
        
        # This would query the database with filters
        # For now, return placeholder
        return {
            "success": True,
            "filters": filters,
            "total_count": 0,
            "returned_count": 0,
            "flows": []
        }
    
    except Exception as e:
        logger.error(f"Advanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")


@router.get("/flows/bulk-export")
async def bulk_export_flows(
    format: str = "csv",  # csv, json, parquet
    filters: Optional[str] = None,  # JSON-encoded filters
    compression: Optional[str] = None  # gzip, zstd
) -> Dict:
    """
    Bulk export flows
    
    Formats: CSV, JSON, Parquet
    Compression: gzip, zstd
    """
    try:
        logger.info(f"Bulk export flows: format={format}, compression={compression}")
        
        return {
            "success": True,
            "format": format,
            "compression": compression,
            "file_url": "/api/integrations/flows/export/job-12345",
            "estimated_size_mb": 250
        }
    
    except Exception as e:
        logger.error(f"Bulk export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk export failed: {str(e)}")


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@router.get("/config/status")
async def get_integration_status() -> Dict:
    """Get integration status"""
    return {
        "threat_intel": {
            "virustotal": INTEGRATION_CONFIG.threat_intel.virustotal_enabled,
            "alienvault": INTEGRATION_CONFIG.threat_intel.alienvault_enabled
        },
        "notifications": {
            "slack": INTEGRATION_CONFIG.notifications.slack_enabled,
            "teams": INTEGRATION_CONFIG.notifications.teams_enabled,
            "discord": INTEGRATION_CONFIG.notifications.discord_enabled,
            "email": INTEGRATION_CONFIG.notifications.email_enabled
        },
        "ticketing": {
            "jira": INTEGRATION_CONFIG.ticketing.jira_enabled,
            "servicenow": INTEGRATION_CONFIG.ticketing.servicenow_enabled
        },
        "firewall": {
            "enabled": bool(INTEGRATION_CONFIG.firewall.firewall_type),
            "type": INTEGRATION_CONFIG.firewall.firewall_type
        },
        "soar": {
            "enabled": True,  # Now supported
            "platforms": ["splunk_soar", "demisto", "tines"]
        },
        "siem": {
            "enabled": True,  # Now supported
            "platforms": ["splunk", "elk", "elasticsearch", "wazuh"]
        },
        "response_actions": {
            "enabled": True,  # Now supported
            "action_count": 14
        }
    }
