"""
Integration Configuration Module
Centralized configuration for all external integrations
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ThreatIntelConfig:
    """Threat Intelligence configuration"""
    virustotal_enabled: bool = False
    virustotal_api_key: str = ""
    virustotal_cache_hours: int = 24
    
    alienvault_enabled: bool = False
    alienvault_api_key: str = ""
    alienvault_cache_hours: int = 24


@dataclass
class NotificationConfig:
    """Notification configuration"""
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: Optional[str] = None
    
    teams_enabled: bool = False
    teams_webhook_url: str = ""
    
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_sender: str = ""
    email_password: str = ""


@dataclass
class TicketingConfig:
    """Ticketing system configuration"""
    jira_enabled: bool = False
    jira_url: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "SEC"
    jira_username: str = "api"
    
    servicenow_enabled: bool = False
    servicenow_url: str = ""
    servicenow_user: str = ""
    servicenow_password: str = ""


@dataclass
class FirewallConfig:
    """Firewall configuration"""
    firewall_type: str = ""  # "pfsense", "fortinet", "checkpoint"
    firewall_host: str = ""
    firewall_api_key: str = ""
    firewall_api_secret: str = ""
    firewall_verify_ssl: bool = True


class INTEGRATION_CONFIG:
    """Master integration configuration"""
    
    threat_intel = ThreatIntelConfig(
        virustotal_enabled=os.getenv("VT_ENABLED", "false").lower() == "true",
        virustotal_api_key=os.getenv("VT_API_KEY", ""),
        alienvault_enabled=os.getenv("AV_ENABLED", "false").lower() == "true",
        alienvault_api_key=os.getenv("AV_API_KEY", "")
    )
    
    notifications = NotificationConfig(
        slack_enabled=os.getenv("SLACK_ENABLED", "false").lower() == "true",
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
        slack_channel=os.getenv("SLACK_CHANNEL"),
        teams_enabled=os.getenv("TEAMS_ENABLED", "false").lower() == "true",
        teams_webhook_url=os.getenv("TEAMS_WEBHOOK_URL", ""),
        discord_enabled=os.getenv("DISCORD_ENABLED", "false").lower() == "true",
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        email_enabled=os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        email_smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        email_smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
        email_sender=os.getenv("EMAIL_SENDER", ""),
        email_password=os.getenv("EMAIL_PASSWORD", "")
    )
    
    ticketing = TicketingConfig(
        jira_enabled=os.getenv("JIRA_ENABLED", "false").lower() == "true",
        jira_url=os.getenv("JIRA_URL", ""),
        jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
        jira_project_key=os.getenv("JIRA_PROJECT_KEY", "SEC"),
        servicenow_enabled=os.getenv("SNOW_ENABLED", "false").lower() == "true",
        servicenow_url=os.getenv("SNOW_URL", ""),
        servicenow_user=os.getenv("SNOW_USER", ""),
        servicenow_password=os.getenv("SNOW_PASSWORD", "")
    )
    
    firewall = FirewallConfig(
        firewall_type=os.getenv("FIREWALL_TYPE", ""),
        firewall_host=os.getenv("FIREWALL_HOST", ""),
        firewall_api_key=os.getenv("FIREWALL_API_KEY", ""),
        firewall_api_secret=os.getenv("FIREWALL_API_SECRET", ""),
        firewall_verify_ssl=os.getenv("FIREWALL_VERIFY_SSL", "true").lower() == "true"
    )
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "INTEGRATION_CONFIG":
        """Create config from dictionary"""
        if "threat_intel" in config_dict:
            cls.threat_intel = ThreatIntelConfig(**config_dict["threat_intel"])
        if "notifications" in config_dict:
            cls.notifications = NotificationConfig(**config_dict["notifications"])
        if "ticketing" in config_dict:
            cls.ticketing = TicketingConfig(**config_dict["ticketing"])
        if "firewall" in config_dict:
            cls.firewall = FirewallConfig(**config_dict["firewall"])
        return cls
    
    @classmethod
    def to_dict(cls) -> Dict:
        """Convert config to dictionary"""
        return {
            "threat_intel": cls.threat_intel.__dict__,
            "notifications": cls.notifications.__dict__,
            "ticketing": cls.ticketing.__dict__,
            "firewall": cls.firewall.__dict__
        }
