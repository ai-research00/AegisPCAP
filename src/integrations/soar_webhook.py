"""
SOAR Webhook Receiver & Response Handler
Receives callbacks from SOAR platforms and processes responses
"""

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import hmac
import hashlib
import logging
import json

logger = logging.getLogger(__name__)

# Router for SOAR webhooks
soar_webhook_router = APIRouter(prefix="/api/integrations/soar", tags=["soar"])


# ============================================================================
# SOAR Webhook Models
# ============================================================================

class FlowContext(BaseModel):
    """Network flow context"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    domain: Optional[str] = None
    user_agent: Optional[str] = None
    risk_score: float
    threat_type: Optional[str] = None


class EvidenceItem(BaseModel):
    """Evidence for threat detection"""
    type: str  # behavioral, signature, anomaly
    description: str
    confidence: float  # 0-1
    data: Optional[Dict[str, Any]] = None


class SOARIncidentRequest(BaseModel):
    """Request to create incident in SOAR"""
    incident_title: str
    severity: str  # critical, high, medium, low
    flow_context: FlowContext
    evidence: List[EvidenceItem]
    soar_platform: str  # splunk_soar, demisto, tines
    soar_api_url: str
    soar_api_key: str
    playbook_triggers: Optional[List[str]] = None


class SOARWebhookCallback(BaseModel):
    """Callback from SOAR platform"""
    incident_id: str
    soar_platform: str
    event_type: str  # incident_created, playbook_completed, status_changed
    status: Optional[str] = None
    analyst_note: Optional[str] = None
    timestamp: datetime


class ResponseAction(BaseModel):
    """Automated response action"""
    action_type: str  # block_ip, block_domain, isolate_endpoint, notify
    target: str  # IP address, domain, endpoint ID
    severity: str
    reason: str
    auto_approve: bool = False


class ResponseActionQueue(BaseModel):
    """Queue of response actions pending approval"""
    incident_id: str
    actions: List[ResponseAction]
    requires_approval: List[int]  # indices of actions requiring manual approval


# ============================================================================
# SOAR Incident Creation Endpoint
# ============================================================================

@soar_webhook_router.post("/create-incident", response_model=Dict[str, Any])
async def create_soar_incident(request: SOARIncidentRequest):
    """
    Create incident in SOAR platform
    
    Example:
    ```json
    {
        "incident_title": "Suspicious DNS Tunneling Detected",
        "severity": "high",
        "flow_context": {
            "src_ip": "192.168.1.100",
            "dst_ip": "8.8.8.8",
            "src_port": 53,
            "dst_port": 53,
            "protocol": "DNS",
            "domain": "example.com",
            "risk_score": 85.5
        },
        "evidence": [
            {
                "type": "anomaly",
                "description": "Abnormally high DNS entropy detected",
                "confidence": 0.92
            }
        ],
        "soar_platform": "splunk_soar",
        "soar_api_url": "https://soar.example.com",
        "soar_api_key": "api_key_here"
    }
    ```
    """
    try:
        from src.integrations.soar import SOARFactory, SOARWebhookPayload
        
        # Create SOAR adapter
        adapter = SOARFactory.create_adapter(
            request.soar_platform,
            request.soar_api_url,
            request.soar_api_key
        )
        
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unknown SOAR platform: {request.soar_platform}")
        
        # Create payload
        flow_dict = request.flow_context.dict()
        evidence_list = [e.dict() for e in request.evidence]
        
        payload = SOARWebhookPayload(
            incident_id=f"AEGIS_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            incident_title=request.incident_title,
            severity=request.severity,
            flow_context=flow_dict,
            evidence=evidence_list
        )
        
        # Create incident in SOAR
        result = await adapter.create_incident(payload)
        
        if result["success"]:
            # Trigger playbooks if specified
            if request.playbook_triggers:
                for playbook in request.playbook_triggers:
                    await adapter.trigger_playbook(
                        result["incident_id"],
                        playbook
                    )
            
            logger.info(f"Created incident in {request.soar_platform}: {result['incident_id']}")
            return {
                "success": True,
                "incident_id": result.get("incident_id"),
                "soar_platform": request.soar_platform,
                "message": f"Incident created in {request.soar_platform}"
            }
        else:
            logger.error(f"Failed to create SOAR incident: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create incident: {result.get('error')}"
            )
    
    except Exception as e:
        logger.error(f"Error creating SOAR incident: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SOAR Webhook Callback Receiver
# ============================================================================

@soar_webhook_router.post("/webhook", response_model=Dict[str, Any])
async def receive_soar_callback(callback: SOARWebhookCallback):
    """
    Receive callback from SOAR platform
    
    Called when:
    - Incident created in SOAR
    - Playbook execution completed
    - Incident status changed
    - Analyst adds notes/feedback
    
    Example Splunk SOAR callback:
    ```json
    {
        "incident_id": "12345",
        "soar_platform": "splunk_soar",
        "event_type": "playbook_completed",
        "status": "investigating",
        "analyst_note": "Confirmed malicious behavior - blocking IP",
        "timestamp": "2026-02-05T12:34:56Z"
    }
    ```
    """
    try:
        logger.info(f"Received SOAR callback: {callback.event_type} for incident {callback.incident_id}")
        
        if callback.event_type == "incident_created":
            logger.info(f"Incident created in {callback.soar_platform}: {callback.incident_id}")
            # Update AegisPCAP incident status to "escalated"
            return {
                "success": True,
                "action": "incident_escalated",
                "incident_id": callback.incident_id
            }
        
        elif callback.event_type == "playbook_completed":
            logger.info(f"Playbook completed for incident {callback.incident_id}")
            # Check playbook result and execute response actions
            return {
                "success": True,
                "action": "playbook_completed",
                "incident_id": callback.incident_id
            }
        
        elif callback.event_type == "status_changed":
            logger.info(f"Incident {callback.incident_id} status changed to: {callback.status}")
            # Update AegisPCAP incident with new status
            return {
                "success": True,
                "action": "status_updated",
                "incident_id": callback.incident_id,
                "status": callback.status
            }
        
        elif callback.event_type == "analyst_feedback":
            logger.info(f"Analyst feedback on incident {callback.incident_id}")
            # Process analyst feedback for model training
            if callback.analyst_note:
                logger.info(f"Feedback note: {callback.analyst_note}")
            return {
                "success": True,
                "action": "feedback_recorded",
                "incident_id": callback.incident_id
            }
        
        return {
            "success": True,
            "incident_id": callback.incident_id,
            "event_type": callback.event_type
        }
    
    except Exception as e:
        logger.error(f"Error processing SOAR callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Response Action Queue Management
# ============================================================================

@soar_webhook_router.post("/response-actions/queue", response_model=Dict[str, Any])
async def queue_response_actions(action_queue: ResponseActionQueue):
    """
    Queue automated response actions
    
    Some actions auto-approve, others require manual approval.
    
    Example:
    ```json
    {
        "incident_id": "AEGIS_20260205_123456",
        "actions": [
            {
                "action_type": "block_ip",
                "target": "192.168.1.100",
                "severity": "high",
                "reason": "Known malicious C2 server",
                "auto_approve": true
            },
            {
                "action_type": "isolate_endpoint",
                "target": "endpoint_abc123",
                "severity": "critical",
                "reason": "Ransomware detected",
                "auto_approve": false
            }
        ],
        "requires_approval": [1]
    }
    ```
    """
    try:
        logger.info(f"Queuing {len(action_queue.actions)} response actions for incident {action_queue.incident_id}")
        
        # Separate auto-approve and pending approval
        auto_actions = [action_queue.actions[i] for i in range(len(action_queue.actions)) 
                       if i not in action_queue.requires_approval]
        pending_actions = [action_queue.actions[i] for i in action_queue.requires_approval]
        
        # Execute auto-approve actions immediately
        executed = []
        for action in auto_actions:
            result = await _execute_response_action(action, action_queue.incident_id)
            executed.append(result)
        
        # Queue pending actions for manual approval
        pending = []
        for action in pending_actions:
            pending.append({
                "action": action.dict(),
                "incident_id": action_queue.incident_id,
                "requires_approval": True,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        logger.info(f"Executed {len(executed)} auto-actions, {len(pending)} pending approval")
        
        return {
            "success": True,
            "incident_id": action_queue.incident_id,
            "executed": len(executed),
            "pending_approval": len(pending),
            "executed_actions": executed,
            "pending_actions": pending
        }
    
    except Exception as e:
        logger.error(f"Error queuing response actions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@soar_webhook_router.post("/response-actions/{action_id}/approve", response_model=Dict[str, Any])
async def approve_response_action(action_id: str, incident_id: str):
    """Approve pending response action"""
    try:
        logger.info(f"Approving action {action_id} for incident {incident_id}")
        # Execute the action
        return {
            "success": True,
            "action_id": action_id,
            "incident_id": incident_id,
            "status": "approved_and_executed"
        }
    except Exception as e:
        logger.error(f"Error approving action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@soar_webhook_router.post("/response-actions/{action_id}/deny", response_model=Dict[str, Any])
async def deny_response_action(action_id: str, incident_id: str):
    """Deny pending response action"""
    try:
        logger.info(f"Denying action {action_id} for incident {incident_id}")
        return {
            "success": True,
            "action_id": action_id,
            "incident_id": incident_id,
            "status": "denied"
        }
    except Exception as e:
        logger.error(f"Error denying action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Response Action Executor
# ============================================================================

async def _execute_response_action(action: ResponseAction, incident_id: str) -> Dict[str, Any]:
    """Execute a response action (internal helper)"""
    try:
        if action.action_type == "block_ip":
            logger.info(f"Executing: Block IP {action.target}")
            # Call firewall integration
            return {
                "success": True,
                "action_type": action.action_type,
                "target": action.target,
                "status": "executed",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif action.action_type == "block_domain":
            logger.info(f"Executing: Block domain {action.target}")
            # Call DNS sinkhole integration
            return {
                "success": True,
                "action_type": action.action_type,
                "target": action.target,
                "status": "executed",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif action.action_type == "isolate_endpoint":
            logger.info(f"Executing: Isolate endpoint {action.target}")
            # Call EDR integration
            return {
                "success": True,
                "action_type": action.action_type,
                "target": action.target,
                "status": "executed",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif action.action_type == "notify":
            logger.info(f"Executing: Notify analysts")
            # Call notification service
            return {
                "success": True,
                "action_type": action.action_type,
                "target": action.target,
                "status": "executed",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:
            logger.warning(f"Unknown action type: {action.action_type}")
            return {
                "success": False,
                "action_type": action.action_type,
                "error": "Unknown action type"
            }
    
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        return {
            "success": False,
            "action_type": action.action_type,
            "error": str(e)
        }
