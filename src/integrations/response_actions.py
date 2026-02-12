"""
Automated Response Actions Executor
Executes automated response actions based on threat severity and detection type
Handles IP blocking, domain blocking, endpoint isolation, notifications, and more
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Response Action Types & Models
# ============================================================================

class ResponseActionType(str, Enum):
    """Types of response actions"""
    BLOCK_IP = "block_ip"
    UNBLOCK_IP = "unblock_ip"
    BLOCK_DOMAIN = "block_domain"
    BLOCK_URL = "block_url"
    ISOLATE_ENDPOINT = "isolate_endpoint"
    RELEASE_ENDPOINT = "release_endpoint"
    DISABLE_ACCOUNT = "disable_account"
    ENABLE_ACCOUNT = "enable_account"
    KILL_PROCESS = "kill_process"
    CAPTURE_TRAFFIC = "capture_traffic"
    NOTIFY = "notify"
    CREATE_TICKET = "create_ticket"
    ESCALATE_INCIDENT = "escalate_incident"
    EXECUTE_PLAYBOOK = "execute_playbook"


class ResponsePriority(str, Enum):
    """Response action priority levels"""
    CRITICAL = "critical"  # Auto-execute immediately
    HIGH = "high"  # Auto-execute, but log
    MEDIUM = "medium"  # Require approval
    LOW = "low"  # Queue for review


class ResponseAction:
    """Represents a response action to be executed"""
    
    def __init__(
        self,
        action_type: ResponseActionType,
        target: str,
        priority: ResponsePriority,
        reason: str,
        incident_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300
    ):
        self.action_id = f"AEGIS_ACTION_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.action_type = action_type
        self.target = target
        self.priority = priority
        self.reason = reason
        self.incident_id = incident_id
        self.parameters = parameters or {}
        self.timeout_seconds = timeout_seconds
        self.created_at = datetime.utcnow()
        self.executed_at: Optional[datetime] = None
        self.status = "pending"  # pending, executing, completed, failed
        self.result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": self.target,
            "priority": self.priority.value,
            "reason": self.reason,
            "incident_id": self.incident_id,
            "parameters": self.parameters,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "result": self.result,
            "error_message": self.error_message
        }


# ============================================================================
# Response Action Executor
# ============================================================================

class ResponseActionExecutor(ABC):
    """Abstract base class for response action executors"""
    
    @abstractmethod
    async def execute(self, action: ResponseAction) -> Dict[str, Any]:
        """Execute a response action"""
        pass
    
    @abstractmethod
    async def validate(self, action: ResponseAction) -> bool:
        """Validate if action can be executed"""
        pass
    
    @abstractmethod
    async def rollback(self, action: ResponseAction) -> Dict[str, Any]:
        """Rollback/undo an action"""
        pass


# ============================================================================
# Network Response Actions
# ============================================================================

class NetworkResponseExecutor(ResponseActionExecutor):
    """Executes network-level response actions"""
    
    def __init__(self, firewall_adapters: Dict[str, Any]):
        self.firewall_adapters = firewall_adapters
    
    async def execute(self, action: ResponseAction) -> Dict[str, Any]:
        """Execute network response action"""
        try:
            if action.action_type == ResponseActionType.BLOCK_IP:
                return await self._block_ip(action)
            elif action.action_type == ResponseActionType.UNBLOCK_IP:
                return await self._unblock_ip(action)
            elif action.action_type == ResponseActionType.BLOCK_DOMAIN:
                return await self._block_domain(action)
            elif action.action_type == ResponseActionType.BLOCK_URL:
                return await self._block_url(action)
            else:
                return {"success": False, "error": "Unknown network action"}
        
        except Exception as e:
            logger.error(f"Error executing network action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, action: ResponseAction) -> bool:
        """Validate network action"""
        if action.action_type == ResponseActionType.BLOCK_IP:
            # Check if valid IP
            return self._is_valid_ip(action.target)
        elif action.action_type == ResponseActionType.BLOCK_DOMAIN:
            return self._is_valid_domain(action.target)
        elif action.action_type == ResponseActionType.BLOCK_URL:
            return action.target.startswith("http")
        return True
    
    async def rollback(self, action: ResponseAction) -> Dict[str, Any]:
        """Rollback network action"""
        try:
            if action.action_type == ResponseActionType.BLOCK_IP:
                # Unblock IP
                unblock_action = ResponseAction(
                    action_type=ResponseActionType.UNBLOCK_IP,
                    target=action.target,
                    priority=action.priority,
                    reason=f"Rollback: {action.reason}",
                    incident_id=action.incident_id
                )
                return await self._unblock_ip(unblock_action)
            
            return {"success": True, "message": "Rollback completed"}
        
        except Exception as e:
            logger.error(f"Error rolling back action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _block_ip(self, action: ResponseAction) -> Dict[str, Any]:
        """Block IP address"""
        try:
            logger.info(f"Blocking IP: {action.target}, Reason: {action.reason}")
            
            # Execute in all configured firewall platforms
            results = {}
            for platform_name, adapter in self.firewall_adapters.items():
                try:
                    result = await adapter.block_ip(
                        action.target,
                        direction=action.parameters.get("direction", "both"),
                        reason=action.reason
                    )
                    results[platform_name] = result
                except Exception as e:
                    results[platform_name] = {"success": False, "error": str(e)}
            
            # Success if at least one firewall executed
            any_success = any(r.get("success") for r in results.values())
            
            return {
                "success": any_success,
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "target": action.target,
                "platforms": results,
                "message": f"IP {action.target} blocking executed"
            }
        
        except Exception as e:
            logger.error(f"Error blocking IP {action.target}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _unblock_ip(self, action: ResponseAction) -> Dict[str, Any]:
        """Unblock IP address"""
        try:
            logger.info(f"Unblocking IP: {action.target}")
            
            results = {}
            for platform_name, adapter in self.firewall_adapters.items():
                if hasattr(adapter, 'unblock_ip'):
                    try:
                        result = await adapter.unblock_ip(action.target)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": True,
                "action_id": action.action_id,
                "target": action.target,
                "message": f"IP {action.target} unblocking executed"
            }
        
        except Exception as e:
            logger.error(f"Error unblocking IP: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _block_domain(self, action: ResponseAction) -> Dict[str, Any]:
        """Block domain"""
        try:
            logger.info(f"Blocking domain: {action.target}")
            
            results = {}
            for platform_name, adapter in self.firewall_adapters.items():
                try:
                    result = await adapter.block_domain(action.target, reason=action.reason)
                    results[platform_name] = result
                except Exception as e:
                    results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()),
                "action_id": action.action_id,
                "target": action.target,
                "platforms": results
            }
        
        except Exception as e:
            logger.error(f"Error blocking domain: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _block_url(self, action: ResponseAction) -> Dict[str, Any]:
        """Block URL pattern"""
        try:
            logger.info(f"Blocking URL: {action.target}")
            
            results = {}
            for platform_name, adapter in self.firewall_adapters.items():
                try:
                    result = await adapter.block_url(action.target, reason=action.reason)
                    results[platform_name] = result
                except Exception as e:
                    results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()),
                "action_id": action.action_id,
                "target": action.target
            }
        
        except Exception as e:
            logger.error(f"Error blocking URL: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Check if valid IP address"""
        parts = ip.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)
    
    @staticmethod
    def _is_valid_domain(domain: str) -> bool:
        """Check if valid domain"""
        return "." in domain and len(domain) > 3


# ============================================================================
# Endpoint Response Actions
# ============================================================================

class EndpointResponseExecutor(ResponseActionExecutor):
    """Executes endpoint-level response actions"""
    
    def __init__(self, edr_adapters: Dict[str, Any]):
        self.edr_adapters = edr_adapters
    
    async def execute(self, action: ResponseAction) -> Dict[str, Any]:
        """Execute endpoint response action"""
        try:
            if action.action_type == ResponseActionType.ISOLATE_ENDPOINT:
                return await self._isolate_endpoint(action)
            elif action.action_type == ResponseActionType.RELEASE_ENDPOINT:
                return await self._release_endpoint(action)
            elif action.action_type == ResponseActionType.KILL_PROCESS:
                return await self._kill_process(action)
            else:
                return {"success": False, "error": "Unknown endpoint action"}
        
        except Exception as e:
            logger.error(f"Error executing endpoint action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, action: ResponseAction) -> bool:
        """Validate endpoint action"""
        return action.target and len(action.target) > 0
    
    async def rollback(self, action: ResponseAction) -> Dict[str, Any]:
        """Rollback endpoint action"""
        try:
            if action.action_type == ResponseActionType.ISOLATE_ENDPOINT:
                release_action = ResponseAction(
                    action_type=ResponseActionType.RELEASE_ENDPOINT,
                    target=action.target,
                    priority=action.priority,
                    reason=f"Rollback: {action.reason}",
                    incident_id=action.incident_id
                )
                return await self._release_endpoint(release_action)
            
            return {"success": True}
        
        except Exception as e:
            logger.error(f"Error rolling back endpoint action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _isolate_endpoint(self, action: ResponseAction) -> Dict[str, Any]:
        """Isolate endpoint from network"""
        try:
            logger.info(f"Isolating endpoint: {action.target}")
            
            # Execute in EDR platforms
            results = {}
            for platform_name, adapter in self.edr_adapters.items():
                if hasattr(adapter, 'isolate_endpoint'):
                    try:
                        result = await adapter.isolate_endpoint(action.target)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()) if results else False,
                "action_id": action.action_id,
                "target": action.target,
                "message": f"Endpoint {action.target} isolation executed"
            }
        
        except Exception as e:
            logger.error(f"Error isolating endpoint: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _release_endpoint(self, action: ResponseAction) -> Dict[str, Any]:
        """Release endpoint back to network"""
        try:
            logger.info(f"Releasing endpoint: {action.target}")
            
            results = {}
            for platform_name, adapter in self.edr_adapters.items():
                if hasattr(adapter, 'release_endpoint'):
                    try:
                        result = await adapter.release_endpoint(action.target)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": True,
                "action_id": action.action_id,
                "target": action.target
            }
        
        except Exception as e:
            logger.error(f"Error releasing endpoint: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _kill_process(self, action: ResponseAction) -> Dict[str, Any]:
        """Kill malicious process on endpoint"""
        try:
            logger.info(f"Killing process on endpoint: {action.target}")
            
            process_id = action.parameters.get("process_id")
            
            results = {}
            for platform_name, adapter in self.edr_adapters.items():
                if hasattr(adapter, 'kill_process'):
                    try:
                        result = await adapter.kill_process(action.target, process_id)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()) if results else False,
                "action_id": action.action_id,
                "target": action.target
            }
        
        except Exception as e:
            logger.error(f"Error killing process: {str(e)}")
            return {"success": False, "error": str(e)}


# ============================================================================
# Identity & Access Response Actions
# ============================================================================

class IdentityResponseExecutor(ResponseActionExecutor):
    """Executes identity/access-level response actions"""
    
    def __init__(self, identity_adapters: Dict[str, Any]):
        self.identity_adapters = identity_adapters
    
    async def execute(self, action: ResponseAction) -> Dict[str, Any]:
        """Execute identity response action"""
        try:
            if action.action_type == ResponseActionType.DISABLE_ACCOUNT:
                return await self._disable_account(action)
            elif action.action_type == ResponseActionType.ENABLE_ACCOUNT:
                return await self._enable_account(action)
            else:
                return {"success": False, "error": "Unknown identity action"}
        
        except Exception as e:
            logger.error(f"Error executing identity action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, action: ResponseAction) -> bool:
        """Validate identity action"""
        return action.target and len(action.target) > 3
    
    async def rollback(self, action: ResponseAction) -> Dict[str, Any]:
        """Rollback identity action"""
        try:
            if action.action_type == ResponseActionType.DISABLE_ACCOUNT:
                enable_action = ResponseAction(
                    action_type=ResponseActionType.ENABLE_ACCOUNT,
                    target=action.target,
                    priority=action.priority,
                    reason=f"Rollback: {action.reason}",
                    incident_id=action.incident_id
                )
                return await self._enable_account(enable_action)
            
            return {"success": True}
        
        except Exception as e:
            logger.error(f"Error rolling back identity action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _disable_account(self, action: ResponseAction) -> Dict[str, Any]:
        """Disable user account"""
        try:
            logger.info(f"Disabling account: {action.target}")
            
            results = {}
            for platform_name, adapter in self.identity_adapters.items():
                if hasattr(adapter, 'disable_account'):
                    try:
                        result = await adapter.disable_account(action.target)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()) if results else False,
                "action_id": action.action_id,
                "target": action.target
            }
        
        except Exception as e:
            logger.error(f"Error disabling account: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _enable_account(self, action: ResponseAction) -> Dict[str, Any]:
        """Enable user account"""
        try:
            logger.info(f"Enabling account: {action.target}")
            
            results = {}
            for platform_name, adapter in self.identity_adapters.items():
                if hasattr(adapter, 'enable_account'):
                    try:
                        result = await adapter.enable_account(action.target)
                        results[platform_name] = result
                    except Exception as e:
                        results[platform_name] = {"success": False, "error": str(e)}
            
            return {
                "success": any(r.get("success") for r in results.values()) if results else True,
                "action_id": action.action_id,
                "target": action.target
            }
        
        except Exception as e:
            logger.error(f"Error enabling account: {str(e)}")
            return {"success": False, "error": str(e)}


# ============================================================================
# Response Action Orchestrator
# ============================================================================

class ResponseActionOrchestrator:
    """Orchestrates and executes response actions across multiple systems"""
    
    def __init__(self):
        self.network_executor: Optional[NetworkResponseExecutor] = None
        self.endpoint_executor: Optional[EndpointResponseExecutor] = None
        self.identity_executor: Optional[IdentityResponseExecutor] = None
        self.action_history: List[ResponseAction] = []
        self.approval_queue: List[ResponseAction] = []
    
    def register_network_executor(self, executor: NetworkResponseExecutor):
        """Register network response executor"""
        self.network_executor = executor
    
    def register_endpoint_executor(self, executor: EndpointResponseExecutor):
        """Register endpoint response executor"""
        self.endpoint_executor = executor
    
    def register_identity_executor(self, executor: IdentityResponseExecutor):
        """Register identity response executor"""
        self.identity_executor = executor
    
    async def execute_action(self, action: ResponseAction) -> Dict[str, Any]:
        """Execute a response action based on its type"""
        try:
            logger.info(f"Processing action: {action.action_type.value} on {action.target}")
            
            # Check if requires approval
            if action.priority == ResponsePriority.MEDIUM or action.priority == ResponsePriority.LOW:
                self.approval_queue.append(action)
                logger.info(f"Action {action.action_id} queued for approval")
                return {
                    "success": True,
                    "action_id": action.action_id,
                    "status": "pending_approval",
                    "message": "Action queued for manual approval"
                }
            
            # Auto-execute critical/high priority actions
            result = None
            
            if action.action_type in [ResponseActionType.BLOCK_IP, ResponseActionType.UNBLOCK_IP,
                                     ResponseActionType.BLOCK_DOMAIN, ResponseActionType.BLOCK_URL]:
                if self.network_executor:
                    result = await self.network_executor.execute(action)
            
            elif action.action_type in [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.RELEASE_ENDPOINT,
                                       ResponseActionType.KILL_PROCESS]:
                if self.endpoint_executor:
                    result = await self.endpoint_executor.execute(action)
            
            elif action.action_type in [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.ENABLE_ACCOUNT]:
                if self.identity_executor:
                    result = await self.identity_executor.execute(action)
            
            if result:
                action.executed_at = datetime.utcnow()
                action.status = "completed" if result.get("success") else "failed"
                action.result = result
                self.action_history.append(action)
                logger.info(f"Action {action.action_id} completed: {result}")
            
            return result or {"success": False, "error": "No executor registered"}
        
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            action.status = "failed"
            action.error_message = str(e)
            self.action_history.append(action)
            return {"success": False, "error": str(e)}
    
    async def approve_action(self, action_id: str) -> Dict[str, Any]:
        """Approve a pending action"""
        try:
            action = next((a for a in self.approval_queue if a.action_id == action_id), None)
            if not action:
                return {"success": False, "error": f"Action {action_id} not found"}
            
            self.approval_queue.remove(action)
            return await self.execute_action(action)
        
        except Exception as e:
            logger.error(f"Error approving action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def reject_action(self, action_id: str) -> Dict[str, Any]:
        """Reject a pending action"""
        try:
            action = next((a for a in self.approval_queue if a.action_id == action_id), None)
            if not action:
                return {"success": False, "error": f"Action {action_id} not found"}
            
            self.approval_queue.remove(action)
            action.status = "rejected"
            self.action_history.append(action)
            logger.info(f"Action {action_id} rejected by analyst")
            
            return {
                "success": True,
                "action_id": action_id,
                "status": "rejected"
            }
        
        except Exception as e:
            logger.error(f"Error rejecting action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def rollback_action(self, action_id: str) -> Dict[str, Any]:
        """Rollback/undo a previously executed action"""
        try:
            action = next((a for a in self.action_history if a.action_id == action_id), None)
            if not action:
                return {"success": False, "error": f"Action {action_id} not found"}
            
            logger.info(f"Rolling back action {action_id}")
            
            # Determine executor and rollback
            if action.action_type in [ResponseActionType.BLOCK_IP, ResponseActionType.UNBLOCK_IP,
                                     ResponseActionType.BLOCK_DOMAIN, ResponseActionType.BLOCK_URL]:
                if self.network_executor:
                    return await self.network_executor.rollback(action)
            
            elif action.action_type in [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.RELEASE_ENDPOINT]:
                if self.endpoint_executor:
                    return await self.endpoint_executor.rollback(action)
            
            elif action.action_type in [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.ENABLE_ACCOUNT]:
                if self.identity_executor:
                    return await self.identity_executor.rollback(action)
            
            return {"success": False, "error": "No executor registered for rollback"}
        
        except Exception as e:
            logger.error(f"Error rolling back action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get all pending actions awaiting approval"""
        return [a.to_dict() for a in self.approval_queue]
    
    def get_action_history(self, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get action history, optionally filtered by incident"""
        if incident_id:
            return [a.to_dict() for a in self.action_history if a.incident_id == incident_id]
        return [a.to_dict() for a in self.action_history]
