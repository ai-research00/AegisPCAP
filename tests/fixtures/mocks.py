"""
Mock external services for testing.

Provides realistic mocked responses for:
- VirusTotal API
- AlienVault OTX API
- Slack Webhook
- Teams Webhook
- Jira API
- ServiceNow API
- Firewall APIs
"""

from unittest.mock import Mock, AsyncMock, MagicMock
import json


class MockVirusTotalResponse:
    """Mock VirusTotal API response."""
    
    @staticmethod
    def malicious_ip():
        """IP with many malicious detections."""
        return {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 45,
                        "suspicious": 5,
                        "undetected": 40
                    },
                    "last_analysis_date": 1706000000,
                    "country": "US"
                }
            }
        }
    
    @staticmethod
    def benign_ip():
        """IP with no malicious detections (Google DNS)."""
        return {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 0,
                        "suspicious": 0,
                        "undetected": 85
                    },
                    "last_analysis_date": 1706000000,
                    "country": "US"
                }
            }
        }
    
    @staticmethod
    def suspicious_ip():
        """IP with some suspicious detections."""
        return {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 5,
                        "suspicious": 15,
                        "undetected": 65
                    },
                    "last_analysis_date": 1706000000,
                    "country": "CN"
                }
            }
        }
    
    @staticmethod
    def malicious_domain():
        """Domain with malicious classification."""
        return {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 30,
                        "suspicious": 10,
                        "undetected": 45
                    },
                    "categories": {
                        "Sophos": "malware",
                        "Kaspersky": "malware"
                    },
                    "last_analysis_date": 1706000000
                }
            }
        }


class MockAlienVaultResponse:
    """Mock AlienVault OTX API response."""
    
    @staticmethod
    def malicious_ip():
        """IP with high reputation score."""
        return {
            "reputation": 65,
            "indicator": "192.168.1.100",
            "type": "IPv4",
            "first_seen": "2026-01-01",
            "last_seen": "2026-02-05"
        }
    
    @staticmethod
    def benign_ip():
        """IP with low reputation score."""
        return {
            "reputation": 0,
            "indicator": "8.8.8.8",
            "type": "IPv4",
            "first_seen": "2020-01-01",
            "last_seen": "2026-02-05"
        }
    
    @staticmethod
    def malicious_domain():
        """Domain with high reputation."""
        return {
            "reputation": 80,
            "indicator": "evil.com",
            "type": "domain",
            "first_seen": "2025-01-01",
            "last_seen": "2026-02-05"
        }


class MockSlackResponse:
    """Mock Slack webhook response."""
    
    @staticmethod
    def success():
        """Successful Slack webhook delivery."""
        return {"ok": True, "channel": "#security-alerts", "ts": "1706000000.000000"}
    
    @staticmethod
    def failure():
        """Failed Slack webhook (invalid URL)."""
        return {"ok": False, "error": "invalid_url"}


class MockJiraResponse:
    """Mock Jira API response."""
    
    @staticmethod
    def create_ticket():
        """Successful ticket creation."""
        return {
            "id": "10001",
            "key": "SEC-123",
            "self": "https://jira.company.com/rest/api/3/issue/10001",
            "fields": {
                "summary": "Security Incident",
                "status": {"name": "To Do"},
                "priority": {"name": "High"}
            }
        }
    
    @staticmethod
    def get_ticket():
        """Get existing ticket."""
        return {
            "id": "10001",
            "key": "SEC-123",
            "fields": {
                "summary": "Security Incident",
                "description": "Investigate suspicious traffic",
                "status": {"name": "In Progress"},
                "assignee": {"name": "analyst@company.com"}
            }
        }
    
    @staticmethod
    def update_ticket():
        """Successful ticket update."""
        return {"id": "10001", "key": "SEC-123"}


class MockServiceNowResponse:
    """Mock ServiceNow API response."""
    
    @staticmethod
    def create_incident():
        """Successful incident creation."""
        return {
            "result": {
                "sys_id": "a1b2c3d4e5f6g7h8",
                "number": "INC0010001",
                "short_description": "Security Incident",
                "state": "1",  # New
                "priority": "2"  # High
            }
        }
    
    @staticmethod
    def get_incident():
        """Get existing incident."""
        return {
            "result": [
                {
                    "sys_id": "a1b2c3d4e5f6g7h8",
                    "number": "INC0010001",
                    "short_description": "Security Incident",
                    "state": "2",  # In Progress
                    "assignment_group": "Security Team"
                }
            ]
        }


class MockFirewallResponse:
    """Mock firewall API responses."""
    
    @staticmethod
    def pfsense_blocked():
        """pfSense IP blocked successfully."""
        return {"status": "success", "message": "IP added to blocklist"}
    
    @staticmethod
    def fortinet_blocked():
        """Fortinet IP blocked successfully."""
        return {"http_status": 200, "version": "v2.0", "status": "success"}
    
    @staticmethod
    def checkpoint_blocked():
        """CheckPoint IP blocked successfully."""
        return {"task-id": "12345", "status": "OK"}


def create_mock_http_client():
    """Create a mock HTTP client for testing."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.patch = AsyncMock()
    mock_client.delete = AsyncMock()
    return mock_client


def create_mock_db():
    """Create a mock database connection."""
    mock_db = MagicMock()
    mock_db.query = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    mock_db.rollback = MagicMock()
    mock_db.close = MagicMock()
    return mock_db


def create_mock_redis():
    """Create a mock Redis client."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=False)
    return mock_redis
