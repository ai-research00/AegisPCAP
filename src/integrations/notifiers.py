"""
Notification Systems Integration
Sends alerts to Slack, Teams, Discord, and email
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

import requests
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AlertMessage:
    """Structured alert message"""
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    source: str  # "AegisPCAP"
    timestamp: datetime
    details: Dict
    action_url: Optional[str] = None


class SlackNotifier:
    """Send alerts to Slack"""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        """
        Initialize Slack notifier
        
        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override (#channel-name)
        """
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_alert(self, alert: AlertMessage) -> bool:
        """
        Send alert to Slack
        
        Args:
            alert: AlertMessage to send
            
        Returns:
            True if successful
        """
        try:
            # Color based on severity
            color_map = {
                "low": "#36a64f",      # Green
                "medium": "#ffa500",   # Orange
                "high": "#ff4444",     # Red
                "critical": "#8b0000"  # Dark red
            }
            color = color_map.get(alert.severity, "#808080")
            
            # Build Slack message
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.isoformat(),
                                "short": False
                            }
                        ],
                        "footer": "AegisPCAP Threat Detection"
                    }
                ]
            }
            
            # Add action button if URL provided
            if alert.action_url:
                payload["attachments"][0]["actions"] = [
                    {
                        "type": "button",
                        "text": "View Details",
                        "url": alert.action_url,
                        "style": "danger"
                    }
                ]
            
            # Override channel if specified
            if self.channel:
                payload["channel"] = self.channel
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def send_alert_async(self, alert: AlertMessage) -> bool:
        """Send alert asynchronously"""
        try:
            color_map = {
                "low": "#36a64f",
                "medium": "#ffa500",
                "high": "#ff4444",
                "critical": "#8b0000"
            }
            color = color_map.get(alert.severity, "#808080")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": False}
                        ],
                        "footer": "AegisPCAP Threat Detection"
                    }
                ]
            }
            
            if alert.action_url:
                payload["attachments"][0]["actions"] = [
                    {"type": "button", "text": "View Details", "url": alert.action_url, "style": "danger"}
                ]
            
            if self.channel:
                payload["channel"] = self.channel
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(f"Slack alert sent: {alert.title}")
                        return True
                    else:
                        logger.error(f"Slack API error: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert (async): {e}")
            return False


class TeamsNotifier:
    """Send alerts to Microsoft Teams"""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Teams notifier
        
        Args:
            webhook_url: Teams webhook URL
        """
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: AlertMessage) -> bool:
        """
        Send alert to Teams
        
        Args:
            alert: AlertMessage to send
            
        Returns:
            True if successful
        """
        try:
            # Color based on severity
            color_map = {
                "low": "28a745",      # Green
                "medium": "ffc107",   # Orange
                "high": "dc3545",     # Red
                "critical": "721c24"  # Dark red
            }
            color = color_map.get(alert.severity, "6c757d")
            
            # Build Teams adaptive card
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": alert.title,
                "themeColor": color,
                "sections": [
                    {
                        "activityTitle": alert.title,
                        "activitySubtitle": alert.source,
                        "text": alert.description,
                        "facts": [
                            {"name": "Severity", "value": alert.severity.upper()},
                            {"name": "Timestamp", "value": alert.timestamp.isoformat()},
                        ]
                    }
                ]
            }
            
            # Add action button if URL provided
            if alert.action_url:
                payload["potentialAction"] = [
                    {
                        "@type": "OpenUri",
                        "name": "View Details",
                        "targets": [
                            {"os": "default", "uri": alert.action_url}
                        ]
                    }
                ]
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Teams alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Teams API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False


class DiscordNotifier:
    """Send alerts to Discord"""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Discord notifier
        
        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: AlertMessage) -> bool:
        """
        Send alert to Discord
        
        Args:
            alert: AlertMessage to send
            
        Returns:
            True if successful
        """
        try:
            # Color based on severity
            color_map = {
                "low": 0x36a64f,      # Green
                "medium": 0xffa500,   # Orange
                "high": 0xff4444,     # Red
                "critical": 0x8b0000  # Dark red
            }
            color = color_map.get(alert.severity, 0x808080)
            
            # Build Discord embed
            payload = {
                "embeds": [
                    {
                        "title": alert.title,
                        "description": alert.description,
                        "color": color,
                        "fields": [
                            {
                                "name": "Severity",
                                "value": alert.severity.upper(),
                                "inline": True
                            },
                            {
                                "name": "Source",
                                "value": alert.source,
                                "inline": True
                            },
                            {
                                "name": "Timestamp",
                                "value": alert.timestamp.isoformat(),
                                "inline": False
                            }
                        ],
                        "footer": {
                            "text": "AegisPCAP Threat Detection"
                        }
                    }
                ]
            }
            
            # Add button if URL provided
            if alert.action_url:
                payload["embeds"][0]["url"] = alert.action_url
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"Discord alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Discord API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False


class EmailNotifier:
    """Send alerts via email"""
    
    def __init__(self, smtp_server: str, smtp_port: int, sender_email: str, sender_password: str):
        """
        Initialize Email notifier
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender password or API token
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_alert(self, alert: AlertMessage, recipient_emails: List[str]) -> bool:
        """
        Send alert via email
        
        Args:
            alert: AlertMessage to send
            recipient_emails: List of recipient email addresses
            
        Returns:
            True if successful
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Build HTML email
            html = f"""
            <html>
                <body>
                    <h2>{alert.title}</h2>
                    <p><strong>Severity:</strong> {alert.severity.upper()}</p>
                    <p><strong>Source:</strong> {alert.source}</p>
                    <p><strong>Timestamp:</strong> {alert.timestamp.isoformat()}</p>
                    <hr>
                    <p>{alert.description}</p>
                    {'<p><a href="' + alert.action_url + '">View Details</a></p>' if alert.action_url else ''}
                </body>
            </html>
            """
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"AegisPCAP Alert: {alert.title}"
            message["From"] = self.sender_email
            message["To"] = ", ".join(recipient_emails)
            
            # Attach HTML
            message.attach(MIMEText(html, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_emails, message.as_string())
            
            logger.info(f"Email alert sent to {len(recipient_emails)} recipients: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
