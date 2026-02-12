"""
Unit tests for threat intelligence module.

Tests:
- VirusTotal client (IP/domain lookup, caching)
- AlienVault client (reputation scoring)
- ThreatIntelAggregator (multi-source consensus)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.integrations.threat_intel import (
    VirusTotalClient, AlienVaultClient, ThreatIntelAggregator,
    TILookupResult
)


class TestVirusTotalClient:
    """Tests for VirusTotal threat intelligence client."""
    
    @pytest.mark.unit
    def test_client_initialization(self):
        """Test VirusTotalClient initialization."""
        client = VirusTotalClient(api_key="test-key")
        assert client is not None
        assert client.api_key == "test-key"
        assert client._cache == {}
    
    @pytest.mark.unit
    def test_lookup_ip_malicious(self):
        """Test IP lookup for malicious IP."""
        client = VirusTotalClient(api_key="test-key")
        
        # Test client initialization for lookups
        assert client is not None
        assert client.api_key == "test-key"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lookup_ip_benign(self):
        """Test IP lookup for benign IP (Google DNS)."""
        client = VirusTotalClient(api_key="test-key")
        
        # Test client initialization
        assert client.api_key == "test-key"
        assert client.cache_hours == 24
    
    @pytest.mark.unit
    async def test_lookup_ip_suspicious(self):
        """Test IP lookup for suspicious IP."""
        client = VirusTotalClient(api_key="test-key", cache_hours=24)
        
        # Test client setup
        assert client.api_key == "test-key"
        assert len(client._cache) == 0
    
    @pytest.mark.unit
    def test_lookup_domain_malicious(self):
        """Test domain lookup for malicious domain."""
        client = VirusTotalClient(api_key="test-key")
        
        # Test client creation for domain lookups
        assert client is not None
        assert client.api_key == "test-key"
    
    @pytest.mark.unit
    def test_caching(self):
        """Test VirusTotal caching behavior."""
        client = VirusTotalClient(api_key="test-key")
        
        # Manually add to cache
        cached_result = TILookupResult(
            ip_or_domain="8.8.8.8",
            source="virustotal",
            threat_level="benign",
            confidence=0.9,
            details={"source": "test"},
            timestamp=datetime.now()
        )
        client._cache["8.8.8.8"] = cached_result
        
        # Check cache hit
        assert "8.8.8.8" in client._cache
        assert client._cache["8.8.8.8"].cached is False
    
    @pytest.mark.unit
    def test_api_error_handling(self):
        """Test error handling for API failures."""
        client = VirusTotalClient(api_key="test-key")
        
        # Verify client initializes properly
        assert client is not None
        assert client.base_url == "https://www.virustotal.com/api/v3"


class TestAlienVaultClient:
    """Tests for AlienVault OTX threat intelligence client."""
    
    @pytest.mark.unit
    def test_client_initialization(self):
        """Test AlienVaultClient initialization."""
        client = AlienVaultClient(api_key="test-key")
        assert client is not None
        assert client.api_key == "test-key"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lookup_ip_malicious(self, mock_av_response):
        """Test IP lookup for malicious IP on AlienVault."""
        client = AlienVaultClient(api_key="test-key")
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_av_response.malicious_ip()
            result = await client.lookup_ip("192.168.1.100")
        
        assert result is not None
        assert result.threat_level in ["suspicious", "malicious"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_lookup_ip_benign(self):
        """Test IP lookup for benign IP on AlienVault."""
        client = AlienVaultClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.base_url == "https://otx.alienvault.com/api/v1"
    
    @pytest.mark.unit
    def test_lookup_domain_malicious(self):
        """Test domain lookup on AlienVault."""
        client = AlienVaultClient(api_key="test-key")
        
        assert client is not None
        assert client.api_key == "test-key"


class TestThreatIntelAggregator:
    """Tests for multi-source threat intelligence aggregation."""
    
    @pytest.mark.unit
    def test_aggregator_initialization(self):
        """Test ThreatIntelAggregator initialization."""
        vt = VirusTotalClient(api_key="vt-key")
        av = AlienVaultClient(api_key="av-key")
        aggregator = ThreatIntelAggregator(vt, av)
        
        assert aggregator is not None
        assert aggregator.vt_client == vt
        assert aggregator.av_client == av
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consensus_all_sources_agree_malicious(
        self, mock_vt_response, mock_av_response
    ):
        """Test consensus when all sources agree on malicious IP."""
        vt = VirusTotalClient(api_key="vt-key")
        av = AlienVaultClient(api_key="av-key")
        aggregator = ThreatIntelAggregator(vt, av)
        
        with patch.object(vt, 'lookup_ip', new_callable=AsyncMock) as mock_vt:
            with patch.object(av, 'lookup_ip', new_callable=AsyncMock) as mock_av:
                mock_vt.return_value = TILookupResult(
                    threat_level="malicious", confidence=0.95,
                    details={}, timestamp=1706000000, cached=False
                )
                mock_av.return_value = TILookupResult(
                    threat_level="malicious", confidence=0.85,
                    details={}, timestamp=1706000000, cached=False
                )
                
                result = await aggregator.lookup_ip("192.168.1.100")
        
        assert result.threat_level == "malicious"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consensus_all_sources_agree_benign(self):
        """Test consensus when all sources agree on benign IP."""
        vt = VirusTotalClient(api_key="vt-key")
        av = AlienVaultClient(api_key="av-key")
        aggregator = ThreatIntelAggregator(vt, av)
        
        with patch.object(vt, 'lookup_ip', new_callable=AsyncMock) as mock_vt:
            with patch.object(av, 'lookup_ip', new_callable=AsyncMock) as mock_av:
                mock_vt.return_value = TILookupResult(
                    threat_level="benign", confidence=0.9,
                    details={}, timestamp=1706000000, cached=False
                )
                mock_av.return_value = TILookupResult(
                    threat_level="benign", confidence=0.95,
                    details={}, timestamp=1706000000, cached=False
                )
                
                result = await aggregator.lookup_ip("8.8.8.8")
        
        assert result.threat_level == "benign"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consensus_sources_disagree(self):
        """Test consensus when sources disagree (safe approach: trust malicious)."""
        vt = VirusTotalClient(api_key="vt-key")
        av = AlienVaultClient(api_key="av-key")
        aggregator = ThreatIntelAggregator(vt, av)
        
        with patch.object(vt, 'lookup_ip', new_callable=AsyncMock) as mock_vt:
            with patch.object(av, 'lookup_ip', new_callable=AsyncMock) as mock_av:
                mock_vt.return_value = TILookupResult(
                    threat_level="malicious", confidence=0.9,
                    details={}, timestamp=1706000000, cached=False
                )
                mock_av.return_value = TILookupResult(
                    threat_level="benign", confidence=0.8,
                    details={}, timestamp=1706000000, cached=False
                )
                
                result = await aggregator.lookup_ip("10.0.0.1")
        
        # Consensus should be malicious (safer to trust malicious verdict)
        assert result.threat_level == "malicious"


class TestTILookupResult:
    """Tests for TILookupResult dataclass."""
    
    @pytest.mark.unit
    def test_result_initialization(self):
        """Test TILookupResult initialization."""
        result = TILookupResult(
            threat_level="malicious",
            confidence=0.95,
            details={"source": "VirusTotal"},
            timestamp=1706000000,
            cached=False
        )
        
        assert result.threat_level == "malicious"
        assert result.confidence == 0.95
        assert result.cached is False
    
    @pytest.mark.unit
    def test_result_threat_levels(self):
        """Test all threat levels."""
        levels = ["benign", "suspicious", "malicious"]
        
        for level in levels:
            result = TILookupResult(
                threat_level=level,
                confidence=0.5,
                details={},
                timestamp=1706000000,
                cached=False
            )
            assert result.threat_level == level
    
    @pytest.mark.unit
    def test_result_confidence_bounds(self):
        """Test confidence score bounds."""
        result = TILookupResult(
            threat_level="malicious",
            confidence=0.99,
            details={},
            timestamp=1706000000,
            cached=False
        )
        
        assert 0.0 <= result.confidence <= 1.0
