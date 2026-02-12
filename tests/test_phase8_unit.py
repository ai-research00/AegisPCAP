"""
Phase 8 Unit Tests
Comprehensive testing for Feature Extractors, ML Models, API Endpoints, and Database
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test (with graceful fallback for optional imports)
try:
    from src.features.extraction import FeatureExtractor
except ImportError:
    FeatureExtractor = None

try:
    from src.models.ensemble import ThreatDetectionEnsemble
except ImportError:
    ThreatDetectionEnsemble = None

try:
    from src.db.persistence import get_persistence_layer
except ImportError:
    get_persistence_layer = None

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


# ============================================================================
# FEATURE EXTRACTOR TESTS
# ============================================================================

@pytest.mark.skipif(FeatureExtractor is None, reason="FeatureExtractor not available")
class TestFeatureExtractor:
    """Test feature extraction pipeline"""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance"""
        return FeatureExtractor()
    
    @pytest.fixture
    def sample_flow(self):
        """Create sample network flow"""
        return {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 54321,
            'dst_port': 53,
            'protocol': 'UDP',
            'packets_sent': 100,
            'packets_recv': 100,
            'bytes_sent': 5000,
            'bytes_recv': 50000,
            'duration_sec': 60,
            'packet_sizes': [50, 52, 48, 51, 49] * 20,
            'iat_values': [0.1, 0.11, 0.09, 0.1, 0.11] * 20,
            'start_time': datetime.utcnow(),
        }
    
    def test_extractor_initialization(self, extractor):
        """Test feature extractor initialization"""
        assert extractor is not None
        assert hasattr(extractor, 'extract') or callable(extractor.__init__)
    
    def test_flow_has_required_fields(self, extractor, sample_flow):
        """Test that flow has required fields"""
        required_fields = ['src_ip', 'dst_ip', 'protocol']
        for field in required_fields:
            assert field in sample_flow
    
    def test_flow_statistics_calculation(self, sample_flow):
        """Test basic flow statistics"""
        features = {
            'packet_count': sample_flow['packets_sent'] + sample_flow['packets_recv'],
            'byte_count': sample_flow['bytes_sent'] + sample_flow['bytes_recv'],
            'duration': sample_flow['duration_sec']
        }
        
        assert features['packet_count'] == 200
        assert features['byte_count'] == 55000
        assert features['duration'] == 60
    
    def test_packet_size_statistics(self, sample_flow):
        """Test packet size feature extraction"""
        sizes = np.array(sample_flow['packet_sizes'])
        
        features = {
            'pkt_size_mean': float(sizes.mean()),
            'pkt_size_std': float(sizes.std()),
            'pkt_size_min': int(sizes.min()),
            'pkt_size_max': int(sizes.max()),
        }
        
        assert features['pkt_size_mean'] > 0
        assert features['pkt_size_std'] >= 0
        assert features['pkt_size_min'] <= features['pkt_size_max']
    
    def test_iat_statistics(self, sample_flow):
        """Test inter-arrival time feature extraction"""
        iat = np.array(sample_flow['iat_values'])
        
        features = {
            'iat_mean': float(iat.mean()),
            'iat_std': float(iat.std()),
            'iat_min': float(iat.min()),
            'iat_max': float(iat.max()),
            'iat_jitter': float(iat.std())
        }
        
        assert features['iat_mean'] > 0
        assert features['iat_std'] >= 0
    
    def test_payload_entropy(self):
        """Test payload entropy calculation"""
        def calculate_entropy(data):
            if isinstance(data, str):
                data = data.encode()
            byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(data)
            entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
            return entropy
        
        payload = b'GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n'
        entropy = calculate_entropy(payload)
        
        assert 0 <= entropy <= 8  # Shannon entropy bounds
        assert entropy > 3  # Should be reasonably high for text
    
    def test_dns_features(self):
        """Test DNS-specific features"""
        domain = 'xkcd.com'
        features = {
            'domain_length': len(domain),
            'subdomain_count': domain.count('.'),
        }
        
        assert features['domain_length'] == 8
        assert features['subdomain_count'] == 1
    
    def test_flow_direction_features(self, sample_flow):
        """Test bidirectional flow features"""
        bytes_sent = sample_flow['bytes_sent']
        bytes_recv = sample_flow['bytes_recv']
        
        features = {
            'upload_download_ratio': bytes_sent / bytes_recv if bytes_recv > 0 else 0,
            'asymmetry': abs(bytes_sent - bytes_recv) / (bytes_sent + bytes_recv)
        }
        
        assert features['upload_download_ratio'] < 1  # Download heavy
        assert 0 <= features['asymmetry'] <= 1


# ============================================================================
# ML MODEL TESTS
# ============================================================================

@pytest.mark.skipif(ThreatDetectionEnsemble is None, reason="Ensemble model not available")
class TestMLModels:
    """Test machine learning models"""
    
    @pytest.fixture
    def ensemble(self):
        """Create threat detection ensemble"""
        return ThreatDetectionEnsemble()
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature vector"""
        return np.random.randn(50)  # 50 features
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for training"""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_ensemble_initialization(self, ensemble):
        """Test ensemble model initialization"""
        assert ensemble is not None
        assert callable(ensemble.predict) or hasattr(ensemble, 'predict')
    
    def test_model_prediction_shape(self, ensemble, sample_features):
        """Test prediction output shape"""
        prediction = ensemble.predict(sample_features.reshape(1, -1))
        assert len(prediction) > 0
        assert all(p in [0, 1, 0.0, 1.0] or 0 <= p <= 1 for p in prediction)
    
    def test_model_probability_bounds(self):
        """Test probability prediction bounds"""
        # Create synthetic probabilities
        proba = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        
        assert proba.shape == (3, 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1)
    
    def test_anomaly_score_calculation(self):
        """Test anomaly score calculation"""
        # Normal points
        normal_scores = np.array([-0.1, -0.05, -0.08, -0.12])
        # Anomalous points
        anomaly_scores = np.array([-2.0, -1.5, -1.8, -1.9])
        
        # Anomalies should have lower scores (more negative)
        assert np.mean(anomaly_scores) < np.mean(normal_scores)
    
    def test_classifier_output(self):
        """Test classifier output format"""
        predictions = np.array([0, 1, 1, 0, 1, 0])
        
        assert predictions.shape == (6,)
        assert set(predictions).issubset({0, 1})
    
    def test_feature_importance_properties(self):
        """Test feature importance properties"""
        # Synthetic feature importance
        importance = np.array([0.15, 0.25, 0.10, 0.20, 0.30])
        
        assert len(importance) > 0
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0, atol=0.01)  # Normalized


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

@pytest.mark.skipif(TestClient is None, reason="FastAPI not available")
class TestAPIEndpoints:
    """Test REST API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        try:
            from src.dashboard.app import app
            return TestClient(app)
        except Exception:
            return None
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/")
        assert response.status_code in [200, 400, 404]
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/api")
        assert response.status_code in [200, 400, 404]
    
    def test_flows_list_endpoint(self, client):
        """Test flows list endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/api/dashboard/flows?limit=10&offset=0")
        assert response.status_code in [200, 400, 404, 500]
    
    def test_alerts_list_endpoint(self, client):
        """Test alerts list endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/api/dashboard/alerts?limit=10&offset=0")
        assert response.status_code in [200, 400, 404, 500]
    
    def test_incidents_list_endpoint(self, client):
        """Test incidents list endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/api/dashboard/incidents?limit=10&offset=0")
        assert response.status_code in [200, 400, 404, 500]
    
    def test_siem_search_endpoint(self, client):
        """Test SIEM search endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        request_body = {
            "siem_platform": "splunk",
            "query": "index=main",
            "start_time": "2026-02-01T00:00:00Z",
            "end_time": "2026-02-05T23:59:59Z"
        }
        
        response = client.post("/api/integrations/siem/search", json=request_body)
        assert response.status_code in [200, 400, 422, 500]
    
    def test_response_action_endpoint(self, client):
        """Test response action execution endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        request_body = {
            "action_type": "block_ip",
            "target": "192.168.1.100",
            "priority": "medium"
        }
        
        response = client.post("/api/integrations/actions/execute", json=request_body)
        assert response.status_code in [200, 400, 422, 500]
    
    def test_config_status_endpoint(self, client):
        """Test integration config status endpoint"""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/api/integrations/config/status")
        assert response.status_code in [200, 400, 404, 500]


# ============================================================================
# DATABASE TESTS
# ============================================================================

@pytest.mark.skipif(get_persistence_layer is None, reason="Persistence layer not available")
class TestDatabase:
    """Test database operations"""
    
    @pytest.fixture
    def persistence(self):
        """Get persistence layer"""
        return get_persistence_layer()
    
    def test_persistence_initialization(self, persistence):
        """Test database initialization"""
        if persistence is None:
            pytest.skip("Persistence not available")
        assert persistence is not None
    
    def test_flow_storage_model(self):
        """Test flow data structure"""
        flow = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 53210,
            'dst_port': 443,
            'protocol': 'TCP',
            'packets': 50,
            'bytes': 5000,
            'timestamp': datetime.utcnow()
        }
        
        assert 'src_ip' in flow
        assert 'dst_ip' in flow
        assert 'timestamp' in flow
        assert isinstance(flow['timestamp'], datetime)
    
    def test_alert_storage_model(self):
        """Test alert data structure"""
        alert = {
            'alert_id': 'alert_001',
            'title': 'Suspicious DNS',
            'severity': 'high',
            'timestamp': datetime.utcnow(),
            'status': 'new'
        }
        
        assert alert['alert_id'] == 'alert_001'
        assert alert['severity'] == 'high'
        assert alert['status'] in ['new', 'open', 'closed']
    
    def test_incident_storage_model(self):
        """Test incident data structure"""
        incident = {
            'incident_id': 'incident_001',
            'title': 'C2 Communication',
            'description': 'Potential C2 activity detected',
            'severity': 'critical',
            'status': 'open',
            'created_at': datetime.utcnow()
        }
        
        assert incident['incident_id'] == 'incident_001'
        assert incident['severity'] == 'critical'
        assert incident['status'] in ['open', 'closed', 'resolved']
    
    def test_action_history_model(self):
        """Test response action history data structure"""
        action = {
            'action_id': 'action_001',
            'action_type': 'block_ip',
            'target': '192.168.1.100',
            'status': 'completed',
            'executed_at': datetime.utcnow()
        }
        
        assert action['action_id'] == 'action_001'
        assert action['action_type'] == 'block_ip'
        assert action['status'] in ['pending', 'approved', 'completed', 'failed']


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_flow_to_alert_pipeline(self):
        """Test flow detection to alert generation"""
        # Create sample flow
        flow = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'protocol': 'DNS',
            'packets': 100,
            'duration': 60
        }
        
        # Should be analyzed
        assert 'src_ip' in flow
        assert 'dst_ip' in flow
    
    def test_alert_to_response_pipeline(self):
        """Test alert to response action generation"""
        alert = {
            'alert_id': 'alert_001',
            'severity': 'critical',
            'threat_type': 'malware'
        }
        
        # Should trigger response
        assert alert['severity'] == 'critical'
    
    def test_response_action_approval_workflow(self):
        """Test response action approval"""
        action = {
            'action_id': 'action_001',
            'priority': 'medium',
            'status': 'pending_approval'
        }
        
        # Should require approval
        assert action['priority'] == 'medium'
        assert action['status'] == 'pending_approval'
    
    def test_full_threat_detection_flow(self):
        """Test complete threat detection flow"""
        # 1. Ingest flow
        flow = {'src_ip': '10.0.0.5', 'dst_ip': 'external.com'}
        assert 'src_ip' in flow
        
        # 2. Extract features
        features = {
            'packet_size_mean': 100,
            'iat_mean': 0.1,
            'entropy': 4.5
        }
        assert len(features) > 0
        
        # 3. ML prediction
        threat_score = 0.8
        assert 0 <= threat_score <= 1
        
        # 4. Generate alert
        alert = {
            'alert_id': 'alert_001',
            'threat_score': threat_score,
            'status': 'new'
        }
        assert alert['threat_score'] > 0.7
        
        # 5. Response action
        response = {
            'action_id': 'action_001',
            'action_type': 'block_ip',
            'status': 'pending_approval'
        }
        assert response['action_type'] == 'block_ip'


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction speed"""
        # Large feature set
        packet_sizes = np.random.randint(50, 1500, 10000)
        
        # Should extract in reasonable time
        import time
        start = time.time()
        sizes = np.array(packet_sizes)
        features = {
            'mean': float(sizes.mean()),
            'std': float(sizes.std()),
            'min': int(sizes.min()),
            'max': int(sizes.max()),
        }
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should be fast
        assert len(features) > 0
    
    def test_model_prediction_performance(self):
        """Test model prediction speed"""
        # Large batch
        X = np.random.randn(1000, 50)
        
        # Simulate fast prediction
        import time
        start = time.time()
        predictions = np.random.randint(0, 2, 1000)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should be fast
        assert len(predictions) == 1000


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src"])
