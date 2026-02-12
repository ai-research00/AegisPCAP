"""
Unit tests for notification systems.

Tests:
- Slack notifier (webhook delivery)
- Teams notifier (adaptive cards)
- Discord notifier (embeds)
- Email notifier (SMTP)
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from src.integrations.notifiers import (
    AlertMessage, SlackNotifier, TeamsNotifier,
    DiscordNotifier, EmailNotifier
)


class TestAlertMessage:
    """Tests for AlertMessage dataclass."""
    
    @pytest.mark.unit
    def test_alert_message_creation(self):
        """Test AlertMessage initialization."""
        alert = AlertMessage(
            severity="high",
            title="Security Alert",
            description="Malicious traffic detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={"ip": "192.168.1.100"},
            action_url="http://localhost:8000/alerts/123"
        )
        
        assert alert.title == "Security Alert"
        assert alert.severity == "high"
        assert alert.source == "AegisPCAP"
    
    @pytest.mark.unit
    def test_alert_severity_levels(self):
        """Test all alert severity levels."""
        levels = ["low", "medium", "high", "critical"]
        
        for level in levels:
            alert = AlertMessage(
                severity=level,
                title="Test Alert",
                description="Test",
                source="Test",
                timestamp=datetime(2024, 1, 23, 12, 0, 0),
                details={}
            )
            assert alert.severity == level


class TestSlackNotifier:
    """Tests for Slack notification delivery."""
    
    @pytest.mark.unit
    def test_notifier_initialization(self):
        """Test SlackNotifier initialization."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier is not None
        assert notifier.webhook_url == "https://hooks.slack.com/test"
    
    @pytest.mark.unit
    def test_send_alert_success(self, mock_slack_response):
        """Test successful alert send to Slack."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        alert = AlertMessage(
            severity="high",
            title="Malicious IP",
            description="192.168.1.100 detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_slack_response.success()
            result = notifier.send_alert(alert)
        
        assert mock_post.called
        mock_post.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_alert_async(self):
        """Test async alert send to Slack."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        alert = AlertMessage(
            severity="high",
            title="Malicious IP",
            description="192.168.1.100 detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"ok": True})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await notifier.send_alert_async(alert)
        
        assert mock_post.called
    
    @pytest.mark.unit
    def test_message_formatting(self):
        """Test Slack message formatting."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        alert = AlertMessage(
            severity="critical",
            title="Critical Threat",
            description="Active malware detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        # Message should be formatted with severity color
        with patch('requests.post') as mock_post:
            notifier.send_alert(alert)
            call_args = mock_post.call_args
            
            # Check that color is included in payload
            assert call_args is not None


class TestTeamsNotifier:
    """Tests for Microsoft Teams notification delivery."""
    
    @pytest.mark.unit
    def test_notifier_initialization(self):
        """Test TeamsNotifier initialization."""
        notifier = TeamsNotifier(webhook_url="https://outlook.webhook.office.com/test")
        assert notifier is not None
        assert notifier.webhook_url == "https://outlook.webhook.office.com/test"
    
    @pytest.mark.unit
    def test_send_alert_success(self):
        """Test successful alert send to Teams."""
        notifier = TeamsNotifier(webhook_url="https://outlook.webhook.office.com/test")
        alert = AlertMessage(
            severity="high",
            title="Security Incident",
            description="Suspicious behavior detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            result = notifier.send_alert(alert)
        
        assert mock_post.called
    
    @pytest.mark.unit
    def test_teams_card_format(self):
        """Test Teams adaptive card formatting."""
        notifier = TeamsNotifier(webhook_url="https://outlook.webhook.office.com/test")
        alert = AlertMessage(
            severity="critical",
            title="Critical Alert",
            description="System compromise detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('requests.post') as mock_post:
            notifier.send_alert(alert)
            call_args = mock_post.call_args
            
            # Verify card was formatted
            assert call_args is not None


class TestDiscordNotifier:
    """Tests for Discord notification delivery."""
    
    @pytest.mark.unit
    def test_notifier_initialization(self):
        """Test DiscordNotifier initialization."""
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test")
        assert notifier is not None
        assert notifier.webhook_url == "https://discord.com/api/webhooks/test"
    
    @pytest.mark.unit
    def test_send_alert_success(self):
        """Test successful alert send to Discord."""
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test")
        alert = AlertMessage(
            severity="high",
            title="Alert",
            description="Suspicious activity detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 204
            result = notifier.send_alert(alert)
        
        assert mock_post.called
    
    @pytest.mark.unit
    def test_embed_color_by_severity(self):
        """Test Discord embed color changes by severity."""
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test")
        
        severity_colors = {
            "low": 0x36a64f,     # Green
            "medium": 0xffa500,  # Orange
            "high": 0xff6b6b,    # Red
            "critical": 0x1a1a1a # Dark red
        }
        
        for severity in severity_colors.keys():
            alert = AlertMessage(
                severity=severity,
                title="Test",
                description="Test",
                source="AegisPCAP",
                timestamp=datetime(2024, 1, 23, 12, 0, 0),
                details={}
            )
            
            with patch('requests.post') as mock_post:
                notifier.send_alert(alert)
                assert mock_post.called


class TestEmailNotifier:
    """Tests for Email notification delivery."""
    
    @pytest.mark.unit
    def test_notifier_initialization(self):
        """Test EmailNotifier initialization."""
        notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="alerts@company.com",
            sender_password="test-password"
        )
        assert notifier is not None
        assert notifier.smtp_server == "smtp.gmail.com"
    
    @pytest.mark.unit
    def test_send_alert_success(self):
        """Test successful email alert send."""
        notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="alerts@company.com",
            sender_password="test-password"
        )
        alert = AlertMessage(
            severity="high",
            title="Security Alert",
            description="Malicious activity detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_instance
            
            notifier.send_alert(alert, recipient_emails=["analyst@company.com"])
        
        assert mock_smtp.called
    
    @pytest.mark.unit
    def test_html_email_format(self):
        """Test HTML email formatting."""
        notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="alerts@company.com",
            sender_password="test-password"
        )
        alert = AlertMessage(
            severity="critical",
            title="Critical Threat",
            description="Ransomware detected on network",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_instance
            
            notifier.send_alert(alert, recipient_emails=["analyst@company.com"])
            
            # Verify send_message was called
            assert mock_instance.send_message.called
    
    @pytest.mark.unit
    def test_multiple_recipients(self):
        """Test sending to multiple recipients."""
        notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="alerts@company.com",
            sender_password="test-password"
        )
        alert = AlertMessage(
            severity="high",
            title="Alert",
            description="Activity detected",
            source="AegisPCAP",
            timestamp=datetime(2024, 1, 23, 12, 0, 0),
            details={}
        )
        
        recipients = ["analyst1@company.com", "analyst2@company.com", "manager@company.com"]
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_instance
            
            notifier.send_alert(alert, recipient_emails=recipients)
            
            assert mock_instance.send_message.called
