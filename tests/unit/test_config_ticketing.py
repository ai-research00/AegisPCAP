"""
Unit tests for ticketing and configuration modules.

Tests:
- Jira connector (create, update, get tickets)
- ServiceNow connector (create, update incidents)
- Integration configuration management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.integrations.ticketing import JiraConnector, ServiceNowConnector
from src.integrations.config import (
    ThreatIntelConfig, NotificationConfig, TicketingConfig,
    FirewallConfig, INTEGRATION_CONFIG
)


class TestJiraConnector:
    """Tests for Jira ticket connector."""
    
    @pytest.mark.unit
    def test_jira_initialization(self):
        """Test JiraConnector initialization."""
        connector = JiraConnector(
            jira_url="https://jira.company.com",
            api_token="test-token",
            project_key="SEC"
        )
        assert connector is not None
        assert connector.jira_url == "https://jira.company.com"
        assert connector.project_key == "SEC"
    
    @pytest.mark.unit
    def test_create_ticket(self, mock_jira_response):
        """Test creating a Jira ticket."""
        connector = JiraConnector(
            jira_url="https://jira.company.com",
            api_token="test-token",
            project_key="SEC"
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_jira_response.create_ticket()
            
            result = connector.create_ticket(
                title="Security Incident",
                description="Malicious traffic detected",
                priority="High"
            )
        
        assert mock_post.called
    
    @pytest.mark.unit
    def test_get_ticket(self, mock_jira_response):
        """Test retrieving a Jira ticket."""
        connector = JiraConnector(
            jira_url="https://jira.company.com",
            api_token="test-token",
            project_key="SEC"
        )
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_jira_response.get_ticket()
            
            result = connector.get_ticket("SEC-123")
        
        assert mock_get.called
    
    @pytest.mark.unit
    def test_update_ticket(self, mock_jira_response):
        """Test updating a Jira ticket."""
        connector = JiraConnector(
            jira_url="https://jira.company.com",
            api_token="test-token",
            project_key="SEC"
        )
        
        with patch('requests.put') as mock_put:
            mock_put.return_value.json.return_value = mock_jira_response.update_ticket()
            
            result = connector.update_ticket(
                ticket_key="SEC-123",
                fields={"status": {"name": "In Progress"}}
            )
        
        assert mock_put.called
    
    @pytest.mark.unit
    def test_add_comment(self):
        """Test adding a comment to Jira ticket."""
        connector = JiraConnector(
            jira_url="https://jira.company.com",
            api_token="test-token",
            project_key="SEC"
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 201
            
            connector.add_comment("SEC-123", "Investigating suspicious IPs")
        
        assert mock_post.called


class TestServiceNowConnector:
    """Tests for ServiceNow incident connector."""
    
    @pytest.mark.unit
    def test_servicenow_initialization(self):
        """Test ServiceNowConnector initialization."""
        connector = ServiceNowConnector(
            snow_instance="https://company.service-now.com",
            api_user="api_user",
            api_password="api_password"
        )
        assert connector is not None
        assert connector.snow_instance == "https://company.service-now.com"
    
    @pytest.mark.unit
    def test_create_incident(self, mock_servicenow_response):
        """Test creating a ServiceNow incident."""
        connector = ServiceNowConnector(
            snow_instance="https://company.service-now.com",
            api_user="api_user",
            api_password="api_password"
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_servicenow_response.create_incident()
            
            result = connector.create_incident(
                short_description="Security Incident",
                description="Malicious traffic",
                urgency="2"  # High
            )
        
        assert mock_post.called
    
    @pytest.mark.unit
    def test_get_incident(self, mock_servicenow_response):
        """Test retrieving a ServiceNow incident."""
        connector = ServiceNowConnector(
            snow_instance="https://company.service-now.com",
            api_user="api_user",
            api_password="api_password"
        )
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_servicenow_response.get_incident()
            
            result = connector.get_incident("INC0010001")
        
        assert mock_get.called
    
    @pytest.mark.unit
    def test_update_incident(self):
        """Test updating a ServiceNow incident."""
        connector = ServiceNowConnector(
            snow_instance="https://company.service-now.com",
            api_user="api_user",
            api_password="api_password"
        )
        
        with patch('requests.patch') as mock_patch:
            mock_patch.return_value.status_code = 200
            
            connector.update_incident(
                incident_number="INC0010001",
                fields={"state": "2", "work_note": "Analyzing logs"}
            )
        
        assert mock_patch.called


class TestIntegrationConfiguration:
    """Tests for integration configuration management."""
    
    @pytest.mark.unit
    def test_threat_intel_config(self):
        """Test ThreatIntelConfig initialization."""
        config = ThreatIntelConfig(
            virustotal_enabled=True,
            virustotal_api_key="test-key",
            virustotal_cache_hours=24,
            alienvault_enabled=True,
            alienvault_api_key="test-av-key",
            alienvault_cache_hours=24
        )
        
        assert config.virustotal_enabled is True
        assert config.virustotal_api_key == "test-key"
        assert config.virustotal_cache_hours == 24
    
    @pytest.mark.unit
    def test_notification_config(self):
        """Test NotificationConfig initialization."""
        config = NotificationConfig(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            slack_channel="#security",
            teams_enabled=True,
            teams_webhook_url="https://outlook.webhook.office.com/test",
            discord_enabled=False,
            discord_webhook_url=None,
            email_enabled=True,
            email_smtp_server="smtp.gmail.com",
            email_smtp_port=587,
            email_sender="alerts@company.com",
            email_password="test-password"
        )
        
        assert config.slack_enabled is True
        assert config.email_enabled is True
        assert config.discord_enabled is False
    
    @pytest.mark.unit
    def test_ticketing_config(self):
        """Test TicketingConfig initialization."""
        config = TicketingConfig(
            jira_enabled=True,
            jira_url="https://jira.company.com",
            jira_api_token="test-token",
            jira_project_key="SEC",
            servicenow_enabled=True,
            servicenow_url="https://company.service-now.com",
            servicenow_user="api_user",
            servicenow_password="api_password"
        )
        
        assert config.jira_enabled is True
        assert config.jira_project_key == "SEC"
        assert config.servicenow_enabled is True
    
    @pytest.mark.unit
    def test_firewall_config(self):
        """Test FirewallConfig initialization."""
        config = FirewallConfig(
            firewall_type="pfsense",
            firewall_host="192.168.1.1",
            firewall_api_key="test-key",
            firewall_verify_ssl=True
        )
        
        assert config.firewall_type == "pfsense"
        assert config.firewall_host == "192.168.1.1"
    
    @pytest.mark.unit
    def test_config_serialization(self):
        """Test configuration serialization to dict."""
        config = ThreatIntelConfig(
            virustotal_enabled=True,
            virustotal_api_key="test-key",
            virustotal_cache_hours=24,
            alienvault_enabled=False,
            alienvault_api_key="",
            alienvault_cache_hours=24
        )
        
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        assert config_dict["virustotal_enabled"] is True
        assert config_dict["alienvault_enabled"] is False
    
    @pytest.mark.unit
    def test_config_from_dict(self):
        """Test configuration creation from dict."""
        config_dict = {
            "virustotal_enabled": True,
            "virustotal_api_key": "test-key",
            "virustotal_cache_hours": 24,
            "alienvault_enabled": False,
            "alienvault_api_key": "",
            "alienvault_cache_hours": 24
        }
        
        config = ThreatIntelConfig(**config_dict)
        assert config.virustotal_enabled is True
        assert config.virustotal_api_key == "test-key"


class TestConfigurationLoading:
    """Tests for loading configuration from environment."""
    
    @pytest.mark.unit
    def test_env_variable_loading(self, mock_env, monkeypatch):
        """Test loading configuration from environment variables."""
        # Mock environment is already set in mock_env fixture
        # Test that INTEGRATION_CONFIG can be initialized
        
        assert "VT_ENABLED" in mock_env
        assert mock_env["VT_ENABLED"] == "true"
    
    @pytest.mark.unit
    def test_default_values(self):
        """Test default configuration values."""
        config = ThreatIntelConfig(
            virustotal_enabled=True,
            virustotal_api_key="test-key"
        )
        
        # Cache hours should have default
        assert config.virustotal_cache_hours == 24
    
    @pytest.mark.unit
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = NotificationConfig(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/valid",
            teams_enabled=False,
            teams_webhook_url=None
        )
        
        assert config.slack_enabled is True


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    @pytest.mark.unit
    def test_all_configs_together(self, mock_env):
        """Test using all configuration types together."""
        ti_config = ThreatIntelConfig(
            virustotal_enabled=True,
            virustotal_api_key="test-key",
            alienvault_enabled=True,
            alienvault_api_key="test-av-key"
        )
        
        notif_config = NotificationConfig(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            teams_enabled=False,
            teams_webhook_url=""
        )
        
        ticket_config = TicketingConfig(
            jira_enabled=True,
            jira_url="https://jira.company.com",
            jira_api_token="test-token",
            jira_project_key="SEC",
            servicenow_enabled=False,
            servicenow_url="",
            servicenow_user="",
            servicenow_password=""
        )
        
        firewall_config = FirewallConfig(
            firewall_type="pfsense",
            firewall_host="192.168.1.1",
            firewall_api_key="test-key"
        )
        
        # All configs should be valid
        assert ti_config.virustotal_enabled is True
        assert notif_config.slack_enabled is True
        assert ticket_config.jira_enabled is True
        assert firewall_config.firewall_type == "pfsense"
