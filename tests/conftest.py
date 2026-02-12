"""Shared pytest fixtures and conftest for all tests."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Generator

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.fixtures.mocks import (
    MockVirusTotalResponse, MockAlienVaultResponse,
    MockSlackResponse, MockJiraResponse, MockServiceNowResponse,
    MockFirewallResponse, create_mock_http_client,
    create_mock_db, create_mock_redis
)


# ============================================================================
# ASYNCIO FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# EXTERNAL SERVICE MOCKS
# ============================================================================

@pytest.fixture
def mock_vt_response():
    """Fixture providing VirusTotal mock responses."""
    return MockVirusTotalResponse()


@pytest.fixture
def mock_av_response():
    """Fixture providing AlienVault OTX mock responses."""
    return MockAlienVaultResponse()


@pytest.fixture
def mock_slack_response():
    """Fixture providing Slack mock responses."""
    return MockSlackResponse()


@pytest.fixture
def mock_jira_response():
    """Fixture providing Jira mock responses."""
    return MockJiraResponse()


@pytest.fixture
def mock_servicenow_response():
    """Fixture providing ServiceNow mock responses."""
    return MockServiceNowResponse()


@pytest.fixture
def mock_firewall_response():
    """Fixture providing firewall mock responses."""
    return MockFirewallResponse()


# ============================================================================
# HTTP CLIENT MOCKS
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Fixture providing a mocked HTTP client."""
    return create_mock_http_client()


@pytest.fixture
def mock_aiohttp_session(mock_http_client):
    """Fixture providing a mocked aiohttp session."""
    with patch('aiohttp.ClientSession', return_value=mock_http_client):
        yield mock_http_client


@pytest.fixture
def mock_requests_session():
    """Fixture providing a mocked requests session."""
    mock_session = Mock()
    mock_session.get = Mock()
    mock_session.post = Mock()
    mock_session.patch = Mock()
    mock_session.delete = Mock()
    return mock_session


# ============================================================================
# DATABASE MOCKS
# ============================================================================

@pytest.fixture
def mock_db():
    """Fixture providing a mocked database connection."""
    return create_mock_db()


@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    return create_mock_redis()


@pytest.fixture
def mock_postgres_connection():
    """Fixture providing a mocked PostgreSQL connection."""
    mock_conn = Mock()
    mock_conn.cursor = Mock()
    mock_conn.execute = Mock()
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()
    return mock_conn


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables for testing."""
    env_vars = {
        "VT_ENABLED": "true",
        "VT_API_KEY": "test-vt-key",
        "AV_ENABLED": "true",
        "AV_API_KEY": "test-av-key",
        "SLACK_ENABLED": "true",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        "JIRA_ENABLED": "true",
        "JIRA_URL": "https://jira.test.com",
        "JIRA_API_TOKEN": "test-jira-token",
        "JIRA_PROJECT_KEY": "TEST",
        "FIREWALL_ENABLED": "true",
        "FIREWALL_TYPE": "pfsense",
        "FIREWALL_HOST": "192.168.1.1",
        "FIREWALL_API_KEY": "test-fw-key",
        "DATABASE_URL": "postgresql://test:test@localhost/aegispcap_test",
        "REDIS_URL": "redis://localhost:6379/1",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_ip_addresses():
    """Fixture providing sample IP addresses for testing."""
    return {
        "benign": "8.8.8.8",
        "suspicious": "192.168.1.100",
        "malicious": "10.0.0.1",
        "private": "192.168.0.1",
        "invalid": "999.999.999.999"
    }


@pytest.fixture
def sample_domains():
    """Fixture providing sample domains for testing."""
    return {
        "benign": "google.com",
        "suspicious": "suspicious.test.com",
        "malicious": "malware.site",
        "phishing": "paypa1.fake"
    }


@pytest.fixture
def sample_alert_message():
    """Fixture providing a sample alert message."""
    return {
        "title": "Malicious IP Detected",
        "description": "IP 192.168.1.100 detected C2 communication",
        "severity": "high",
        "source": "AegisPCAP",
        "timestamp": 1706000000,
        "details": {
            "ip": "192.168.1.100",
            "threat_level": "malicious",
            "confidence": 0.95
        }
    }


@pytest.fixture
def sample_ticket_data():
    """Fixture providing sample ticket data."""
    return {
        "title": "Security Incident Investigation",
        "description": "Investigate suspicious network traffic",
        "system": "jira",
        "priority": "High",
        "labels": ["security", "network", "investigation"]
    }


@pytest.fixture
def sample_pcap_flow():
    """Fixture providing a sample PCAP flow."""
    return {
        "flow_id": "192.168.1.100-8.8.8.8-6-53",
        "src_ip": "192.168.1.100",
        "dst_ip": "8.8.8.8",
        "protocol": 6,  # TCP
        "src_port": 12345,
        "dst_port": 53,
        "packet_count": 100,
        "byte_count": 50000,
        "duration": 10.5,
        "start_time": 1706000000,
        "end_time": 1706000010.5,
    }


# ============================================================================
# FASTAPI TEST CLIENT
# ============================================================================

@pytest.fixture
def api_client():
    """Fixture providing a FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.dashboard.app import app
    
    return TestClient(app)


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically clean up after each test."""
    yield
    # Clean up code here if needed


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture providing a temporary directory for tests."""
    return tmp_path
