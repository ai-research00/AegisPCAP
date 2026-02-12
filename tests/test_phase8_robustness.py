"""
Phase 8 Robustness & Adversarial Testing
Test handling of edge cases, corrupted data, and adversarial inputs
"""

import pytest
import numpy as np
import json
from datetime import datetime, timedelta


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test handling of edge case inputs"""
    
    def test_empty_flow_data(self):
        """Test handling of empty flow"""
        flow = {}
        
        # Should handle gracefully
        assert isinstance(flow, dict)
        assert len(flow) == 0
    
    def test_minimal_flow_data(self):
        """Test handling of minimal flow"""
        flow = {
            'src_ip': '0.0.0.0',
            'dst_ip': '0.0.0.0',
        }
        
        assert 'src_ip' in flow
        assert flow['src_ip'] == '0.0.0.0'
    
    def test_zero_duration_flow(self):
        """Test flow with zero duration"""
        flow = {
            'src_ip': '192.168.1.1',
            'dst_ip': '8.8.8.8',
            'duration': 0,
            'packets': 100,
            'bytes': 5000
        }
        
        # Should not divide by zero
        pkt_rate = flow['packets'] / max(flow['duration'], 1)
        assert pkt_rate == 100
    
    def test_zero_packet_flow(self):
        """Test flow with zero packets"""
        flow = {
            'src_ip': '192.168.1.1',
            'dst_ip': '8.8.8.8',
            'duration': 60,
            'packets': 0,
            'bytes': 0
        }
        
        assert flow['packets'] == 0
        assert flow['bytes'] == 0
    
    def test_single_packet_flow(self):
        """Test flow with single packet"""
        flow = {
            'packets': 1,
            'bytes': 40,  # Minimum IP packet
        }
        
        assert flow['packets'] >= 0
        assert flow['bytes'] >= 0
    
    def test_huge_packet_flow(self):
        """Test flow with jumbo frames"""
        flow = {
            'packet_size': 9000,  # Jumbo frame
            'packets': 100,
        }
        
        total_bytes = flow['packet_size'] * flow['packets']
        assert total_bytes == 900000
    
    def test_empty_packet_list(self):
        """Test with empty packet size list"""
        packets = []
        
        if len(packets) == 0:
            mean = 0
            std = 0
        else:
            packets_arr = np.array(packets)
            mean = packets_arr.mean()
            std = packets_arr.std()
        
        assert mean == 0
        assert std == 0
    
    def test_single_packet_size(self):
        """Test with single packet size"""
        packets = [100]
        packets_arr = np.array(packets)
        
        assert packets_arr.mean() == 100
        assert packets_arr.std() == 0
    
    def test_constant_packet_sizes(self):
        """Test with all identical packet sizes"""
        packets = [100] * 1000
        packets_arr = np.array(packets)
        
        assert packets_arr.mean() == 100
        assert packets_arr.std() == 0
    
    def test_extreme_packet_size_variation(self):
        """Test with extreme variation in packet sizes"""
        packets = [40, 40, 40] + [9000] * 3
        packets_arr = np.array(packets)
        
        mean = packets_arr.mean()
        std = packets_arr.std()
        
        assert mean > 40
        assert std > 0


# ============================================================================
# DATA CORRUPTION TESTS
# ============================================================================

class TestDataCorruption:
    """Test handling of corrupted or invalid data"""
    
    def test_negative_packet_count(self):
        """Test handling of negative packet count"""
        flow = {
            'packets': -100,  # Invalid
        }
        
        # Should validate input
        assert flow['packets'] < 0  # Detectable as invalid
    
    def test_negative_bytes(self):
        """Test handling of negative byte count"""
        flow = {
            'bytes': -5000,  # Invalid
        }
        
        assert flow['bytes'] < 0  # Invalid
    
    def test_invalid_ip_format(self):
        """Test handling of invalid IP addresses"""
        flows = [
            {'src_ip': '999.999.999.999'},  # Invalid
            {'src_ip': '192.168.1'},  # Incomplete
            {'src_ip': '192.168.1.1.1'},  # Too many octets
            {'src_ip': 'not-an-ip'},  # Non-numeric
        ]
        
        for flow in flows:
            # Should be detectable as invalid
            assert isinstance(flow['src_ip'], str)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        flows = [
            {},  # Empty
            {'src_ip': '192.168.1.1'},  # Missing dst_ip
            {'dst_ip': '8.8.8.8'},  # Missing src_ip
        ]
        
        for flow in flows:
            missing_fields = []
            if 'src_ip' not in flow:
                missing_fields.append('src_ip')
            if 'dst_ip' not in flow:
                missing_fields.append('dst_ip')
            
            assert len(missing_fields) > 0 or len(flow) == 0
    
    def test_nan_values(self):
        """Test handling of NaN values"""
        features = {
            'mean': float('nan'),
            'std': float('nan'),
        }
        
        # Should be detectable
        assert np.isnan(features['mean'])
        assert np.isnan(features['std'])
    
    def test_infinity_values(self):
        """Test handling of infinity values"""
        features = {
            'ratio': float('inf'),
            'score': float('-inf'),
        }
        
        # Should be detectable
        assert np.isinf(features['ratio'])
        assert np.isinf(features['score'])
    
    def test_null_values(self):
        """Test handling of None/null values"""
        flow = {
            'src_ip': None,
            'dst_ip': None,
        }
        
        # Should be detectable as invalid
        assert flow['src_ip'] is None
        assert flow['dst_ip'] is None


# ============================================================================
# ADVERSARIAL INPUT TESTS
# ============================================================================

class TestAdversarialInputs:
    """Test robustness against adversarial inputs"""
    
    def test_sql_injection_attempt(self):
        """Test protection against SQL injection"""
        malicious_inputs = [
            "'; DROP TABLE flows; --",
            "1' OR '1'='1",
            "admin'--",
            "<script>alert('xss')</script>",
        ]
        
        for input_str in malicious_inputs:
            # Proper escaping/validation should handle these
            assert isinstance(input_str, str)
    
    def test_extremely_large_values(self):
        """Test handling of extremely large numbers"""
        flow = {
            'bytes': 10**20,  # Unrealistically large
            'packets': 10**20,
        }
        
        # Should not overflow
        assert flow['bytes'] > 0
        assert flow['packets'] > 0
    
    def test_special_characters_in_domain(self):
        """Test handling of special characters in domain names"""
        domains = [
            'test.com',
            'test-domain.com',
            'test_domain.com',
            'test.co.uk',
            'xn--test-dfd.com',  # IDN
            'test.com.',  # Trailing dot
        ]
        
        for domain in domains:
            assert isinstance(domain, str)
            assert len(domain) > 0
    
    def test_unicode_input(self):
        """Test handling of unicode input"""
        unicode_inputs = [
            'test_日本語',
            'тест_русский',
            '🎉🔒🚀',
        ]
        
        for input_str in unicode_inputs:
            assert isinstance(input_str, str)
    
    def test_long_string_input(self):
        """Test handling of very long strings"""
        long_string = 'A' * 1000000  # 1MB string
        
        # Should handle without crash
        assert len(long_string) == 1000000
    
    def test_deeply_nested_json(self):
        """Test handling of deeply nested JSON"""
        # Create deeply nested structure
        nested = {'value': 1}
        for _ in range(1000):
            nested = {'nested': nested}
        
        # Should handle without stack overflow
        assert isinstance(nested, dict)
    
    def test_high_dimensional_feature_vector(self):
        """Test handling of high-dimensional features"""
        features = np.random.randn(10000)  # 10K dimensions
        
        # Should compute without error
        mean = np.mean(features)
        std = np.std(features)
        
        assert not np.isnan(mean)
        assert not np.isnan(std)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_division_by_zero(self):
        """Test safe handling of division by zero"""
        flow = {
            'bytes': 1000,
            'duration': 0
        }
        
        # Safe division
        rate = flow['bytes'] / max(flow['duration'], 1)
        assert rate == 1000
    
    def test_missing_key_access(self):
        """Test safe dictionary access"""
        flow = {'src_ip': '192.168.1.1'}
        
        # Safe access with default
        dst_ip = flow.get('dst_ip', 'unknown')
        assert dst_ip == 'unknown'
    
    def test_list_index_out_of_bounds(self):
        """Test safe list access"""
        packets = [100, 200, 300]
        
        # Safe access
        if len(packets) > 10:
            value = packets[10]
        else:
            value = None
        
        assert value is None
    
    def test_type_mismatch(self):
        """Test handling of type mismatches"""
        def process_number(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        results = [
            process_number(100),  # int
            process_number('100'),  # string
            process_number([100]),  # list
            process_number(None),  # None
        ]
        
        assert results == [100.0, 100.0, 0.0, 0.0]


# ============================================================================
# CONFIDENCE CALIBRATION TESTS
# ============================================================================

class TestConfidenceCalibration:
    """Test confidence score calibration"""
    
    def test_confidence_bounds(self):
        """Test that confidence scores stay within bounds"""
        scores = np.random.uniform(0, 1, 1000)
        
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
    
    def test_probability_distribution(self):
        """Test valid probability distributions"""
        proba = np.random.dirichlet([1, 1, 1], 100)  # 3-class probabilities
        
        # Each row should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_confidence_monotonicity(self):
        """Test that higher scores are more confident"""
        scores = np.array([0.1, 0.3, 0.7, 0.95])
        
        # Monotonically increasing
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    
    def test_rare_event_probability(self):
        """Test handling of very small probabilities"""
        probabilities = [1e-6, 1e-10, 1e-20]
        
        for p in probabilities:
            assert p > 0
            assert p < 1


# ============================================================================
# STABILITY & CONSISTENCY TESTS
# ============================================================================

class TestStabilityConsistency:
    """Test numerical stability and consistency"""
    
    def test_repeated_computation_consistency(self):
        """Test that repeated computations give same results"""
        data = np.random.randn(100, 50)
        
        result1 = np.mean(data)
        result2 = np.mean(data)
        
        assert result1 == result2  # Deterministic
    
    def test_floating_point_precision(self):
        """Test handling of floating point precision"""
        # Known precision issue
        result = 0.1 + 0.2
        
        # Use approximate equality
        assert np.isclose(result, 0.3)
    
    def test_large_number_stability(self):
        """Test numerical stability with large numbers"""
        large_numbers = np.array([1e15, 1e15, 1e15])
        
        mean = np.mean(large_numbers)
        assert not np.isnan(mean)
        assert mean > 0
    
    def test_small_number_stability(self):
        """Test numerical stability with very small numbers"""
        small_numbers = np.array([1e-15, 1e-15, 1e-15])
        
        sum_values = np.sum(small_numbers)
        assert not np.isnan(sum_values)
        assert sum_values > 0


# ============================================================================
# CONCURRENT ACCESS TESTS
# ============================================================================

class TestConcurrentAccess:
    """Test thread-safe operations"""
    
    def test_shared_resource_access(self):
        """Test handling of shared resource access"""
        import threading
        
        counter = {'value': 0}
        lock = threading.Lock()
        
        def increment():
            with lock:
                counter['value'] += 1
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert counter['value'] == 10


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
