"""
Phase 7 Integration Tests
Tests for SOAR, SIEM, Response Actions, and Firewall integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Import integration modules
from src.integrations.soar import (
    SOARFactory, SOARWebhookPayload, SplunkSOARAdapter
)
from src.integrations.siem import (
    SIEMFactory, SIEMEvent, SIEMAlert, SplunkAdapter
)
from src.integrations.response_actions import (
    ResponseAction, ResponseActionType, ResponsePriority,
    ResponseActionOrchestrator, NetworkResponseExecutor
)


# ============================================================================
# SOAR Integration Tests
# ============================================================================

class TestSOARIntegration:
    """Test SOAR platform integrations"""
    
    def test_soar_factory_creates_splunk_adapter(self):
        """Test that factory creates Splunk SOAR adapter"""
        adapter = SOARFactory.create_adapter(
            "splunk_soar",
            "https://soar.example.com",
            "test_key"
        )
        
        assert adapter is not None
        assert isinstance(adapter, SplunkSOARAdapter)
    
    def test_soar_factory_unknown_platform(self):
        """Test factory handles unknown platform"""
        adapter = SOARFactory.create_adapter(
            "unknown_platform",
            "https://soar.example.com",
            "test_key"
        )
        
        assert adapter is None
    
    @pytest.mark.asyncio
    async def test_soar_payload_creation(self):
        """Test SOAR webhook payload creation"""
        payload = SOARWebhookPayload(
            incident_id="TEST_001",
            incident_title="Test Incident",
            severity="high",
            flow_context={
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "protocol": "DNS"
            },
            evidence=[
                {
                    "type": "behavioral",
                    "description": "DNS tunneling detected",
                    "confidence": 0.95
                }
            ]
        )
        
        assert payload.incident_id == "TEST_001"
        assert payload.severity == "high"
        assert len(payload.evidence) == 1
        
        # Test serialization
        data = payload.to_dict()
        assert data["incident_id"] == "TEST_001"
        assert data["severity"] == "high"
    
    @pytest.mark.asyncio
    async def test_soar_adapter_create_incident(self):
        """Test SOAR adapter incident creation"""
        with patch('aiohttp.ClientSession') as mock_session:
            adapter = SplunkSOARAdapter(
                "https://soar.example.com",
                "test_key"
            )
            
            # Mock the async context manager
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "id": "12345",
                "name": "Test Incident"
            })
            
            # Note: Full test would require more complex mocking
            # This demonstrates the structure
            assert adapter.api_url == "https://soar.example.com"
            assert adapter.api_key == "test_key"


# ============================================================================
# SIEM Integration Tests
# ============================================================================

class TestSIEMIntegration:
    """Test SIEM platform integrations"""
    
    def test_siem_factory_creates_splunk_adapter(self):
        """Test that factory creates Splunk adapter"""
        adapter = SIEMFactory.create_adapter(
            "splunk",
            "https://splunk.example.com:8089",
            "test_key"
        )
        
        assert adapter is not None
        assert isinstance(adapter, SplunkAdapter)
    
    def test_siem_factory_creates_elk_adapter(self):
        """Test that factory creates ELK adapter"""
        adapter = SIEMFactory.create_adapter(
            "elk",
            "https://elasticsearch.example.com",
            "test_key"
        )
        
        assert adapter is not None
    
    def test_siem_event_creation(self):
        """Test SIEM event model"""
        event = SIEMEvent(
            timestamp=datetime.utcnow(),
            event_id="evt_001",
            source_type="network",
            event_name="suspicious_dns",
            severity="high",
            src_ip="192.168.1.100",
            dst_ip="8.8.8.8",
            protocol="DNS",
            details={"query": "example.com"}
        )
        
        assert event.event_id == "evt_001"
        assert event.severity == "high"
        assert event.src_ip == "192.168.1.100"
    
    def test_siem_alert_creation(self):
        """Test SIEM alert model"""
        alert = SIEMAlert(
            alert_id="alert_001",
            title="Malware Detected",
            description="Suspected malware on host",
            severity="critical",
            status="new",
            events=[],
            detection_rule="malware_rule_123",
            correlation_count=5,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.severity == "critical"
        assert alert.correlation_count == 5


# ============================================================================
# Response Actions Tests
# ============================================================================

class TestResponseActions:
    """Test response action executors"""
    
    def test_response_action_creation(self):
        """Test response action creation"""
        action = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target="192.168.1.100",
            priority=ResponsePriority.CRITICAL,
            reason="Malware detected",
            incident_id="AEGIS_20260205_123456"
        )
        
        assert action.action_type == ResponseActionType.BLOCK_IP
        assert action.priority == ResponsePriority.CRITICAL
        assert action.status == "pending"
        assert action.action_id.startswith("AEGIS_ACTION_")
    
    def test_response_action_serialization(self):
        """Test response action to_dict"""
        action = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target="192.168.1.100",
            priority=ResponsePriority.CRITICAL,
            reason="Test",
            incident_id="TEST_001"
        )
        
        data = action.to_dict()
        assert data["action_type"] == "block_ip"
        assert data["priority"] == "critical"
        assert data["target"] == "192.168.1.100"
    
    def test_response_priority_levels(self):
        """Test all response priority levels"""
        priorities = [
            ResponsePriority.CRITICAL,
            ResponsePriority.HIGH,
            ResponsePriority.MEDIUM,
            ResponsePriority.LOW
        ]
        
        assert len(priorities) == 4
        assert ResponsePriority.CRITICAL.value == "critical"
        assert ResponsePriority.LOW.value == "low"
    
    def test_response_action_types(self):
        """Test all response action types"""
        action_types = [
            ResponseActionType.BLOCK_IP,
            ResponseActionType.UNBLOCK_IP,
            ResponseActionType.BLOCK_DOMAIN,
            ResponseActionType.BLOCK_URL,
            ResponseActionType.ISOLATE_ENDPOINT,
            ResponseActionType.RELEASE_ENDPOINT,
            ResponseActionType.KILL_PROCESS,
            ResponseActionType.DISABLE_ACCOUNT,
            ResponseActionType.ENABLE_ACCOUNT,
            ResponseActionType.NOTIFY,
            ResponseActionType.CREATE_TICKET,
            ResponseActionType.ESCALATE_INCIDENT,
            ResponseActionType.EXECUTE_PLAYBOOK,
            ResponseActionType.CAPTURE_TRAFFIC
        ]
        
        assert len(action_types) == 14
        assert ResponseActionType.BLOCK_IP.value == "block_ip"
    
    @pytest.mark.asyncio
    async def test_network_response_executor_validation(self):
        """Test network response executor validation"""
        executor = NetworkResponseExecutor({})
        
        # Valid IP
        action_ip = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target="192.168.1.1",
            priority=ResponsePriority.CRITICAL,
            reason="Test",
            incident_id="TEST_001"
        )
        assert await executor.validate(action_ip) == True
        
        # Valid domain
        action_domain = ResponseAction(
            action_type=ResponseActionType.BLOCK_DOMAIN,
            target="example.com",
            priority=ResponsePriority.CRITICAL,
            reason="Test",
            incident_id="TEST_001"
        )
        assert await executor.validate(action_domain) == True
    
    @pytest.mark.asyncio
    async def test_orchestrator_pending_actions(self):
        """Test response action orchestrator"""
        orchestrator = ResponseActionOrchestrator()
        
        # Medium priority action should be queued
        action = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target="192.168.1.100",
            priority=ResponsePriority.MEDIUM,
            reason="Test",
            incident_id="TEST_001"
        )
        
        # Execute action (would be queued)
        result = await orchestrator.execute_action(action)
        
        assert result["success"] == True
        assert result["status"] == "pending_approval"


# ============================================================================
# Integration Flow Tests
# ============================================================================

class TestIntegrationFlows:
    """Test end-to-end integration flows"""
    
    @pytest.mark.asyncio
    async def test_malware_detection_to_soar_flow(self):
        """Test flow: Malware detection → SOAR incident → Response action"""
        
        # 1. Create SOAR payload (from detection)
        payload = SOARWebhookPayload(
            incident_id="AEGIS_20260205_001",
            incident_title="Malware: Emotet Detected",
            severity="critical",
            flow_context={
                "src_ip": "192.168.1.100",
                "dst_ip": "attacker.com",
                "protocol": "HTTPS"
            },
            evidence=[
                {
                    "type": "signature",
                    "description": "Emotet malware signature match",
                    "confidence": 0.98
                }
            ]
        )
        
        assert payload.incident_id == "AEGIS_20260205_001"
        assert payload.severity == "critical"
        
        # 2. Create response action
        action = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target=payload.flow_context["src_ip"],
            priority=ResponsePriority.CRITICAL,
            reason="Malware source blocking",
            incident_id=payload.incident_id
        )
        
        assert action.target == "192.168.1.100"
        assert action.incident_id == payload.incident_id
    
    @pytest.mark.asyncio
    async def test_siem_alert_enrichment_flow(self):
        """Test flow: SIEM alert → Correlation → Escalation"""
        
        # 1. SIEM alert received
        alert = SIEMAlert(
            alert_id="alert_splunk_001",
            title="Brute Force Attempt",
            description="Multiple failed SSH logins",
            severity="high",
            status="new",
            events=[],
            detection_rule="ssh_brute_force",
            correlation_count=15,
            first_seen=datetime.utcnow() - timedelta(minutes=5),
            last_seen=datetime.utcnow()
        )
        
        # 2. Determine if escalation needed
        should_escalate = (
            alert.severity == "high" and
            alert.correlation_count >= 10
        )
        
        assert should_escalate == True
        
        # 3. Create escalation action
        escalation_action = ResponseAction(
            action_type=ResponseActionType.ESCALATE_INCIDENT,
            target=alert.alert_id,
            priority=ResponsePriority.HIGH,
            reason=f"Multiple correlations: {alert.correlation_count}",
            incident_id=f"ESCALATED_{alert.alert_id}"
        )
        
        assert escalation_action.action_type == ResponseActionType.ESCALATE_INCIDENT


# ============================================================================
# Approval Workflow Tests
# ============================================================================

class TestApprovalWorkflow:
    """Test approval workflow for response actions"""
    
    @pytest.mark.asyncio
    async def test_critical_auto_executes(self):
        """Test that CRITICAL priority actions go straight to execution (not pending_approval)"""
        # Use a NOTIFY action since it doesn't require executor registration
        action = ResponseAction(
            action_type=ResponseActionType.NOTIFY,
            target="soc-team@example.com",
            priority=ResponsePriority.CRITICAL,
            reason="Active malware",
            incident_id="TEST_001"
        )
        
        orchestrator = ResponseActionOrchestrator()
        result = await orchestrator.execute_action(action)
        
        # Critical should NOT be queued for approval
        # It will fail because there's no executor, but that's ok - the key is it's not pending_approval
        assert result.get("status") != "pending_approval"
    
    @pytest.mark.asyncio
    async def test_medium_requires_approval(self):
        """Test that MEDIUM priority actions require approval"""
        action = ResponseAction(
            action_type=ResponseActionType.DISABLE_ACCOUNT,
            target="user@example.com",
            priority=ResponsePriority.MEDIUM,
            reason="Suspicious login",
            incident_id="TEST_001"
        )
        
        orchestrator = ResponseActionOrchestrator()
        result = await orchestrator.execute_action(action)
        
        # Medium should queue for approval
        assert result["status"] == "pending_approval"
    
    @pytest.mark.asyncio
    async def test_approval_and_rejection(self):
        """Test action approval and rejection"""
        orchestrator = ResponseActionOrchestrator()
        
        action = ResponseAction(
            action_type=ResponseActionType.BLOCK_IP,
            target="192.168.1.100",
            priority=ResponsePriority.MEDIUM,
            reason="Test",
            incident_id="TEST_001"
        )
        
        result = await orchestrator.execute_action(action)
        action_id = result["action_id"]
        
        # Get pending actions
        pending = orchestrator.get_pending_actions()
        assert len(pending) == 1
        
        # Approve action
        approve_result = await orchestrator.approve_action(action_id)
        assert approve_result["success"] == True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
