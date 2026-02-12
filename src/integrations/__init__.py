"""
AegisPCAP Integrations Module
Connects AegisPCAP to external platforms: SOAR, SIEM, TI, Notifications, Ticketing
"""

from .threat_intel import VirusTotalClient, AlienVaultClient, ThreatIntelAggregator
from .notifiers import SlackNotifier, TeamsNotifier, DiscordNotifier, EmailNotifier
from .ticketing import JiraConnector, ServiceNowConnector
from .firewall import FirewallConnector
from .config import INTEGRATION_CONFIG

__all__ = [
    # Threat Intelligence
    "VirusTotalClient",
    "AlienVaultClient",
    "ThreatIntelAggregator",
    # Notifications
    "SlackNotifier",
    "TeamsNotifier",
    "DiscordNotifier",
    "EmailNotifier",
    # Ticketing
    "JiraConnector",
    "ServiceNowConnector",
    # Firewall
    "FirewallConnector",
    # Config
    "INTEGRATION_CONFIG",
]
