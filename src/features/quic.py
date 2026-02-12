"""
QUIC/HTTP3 Features - QUIC-specific anomaly detection
"""
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class QUICFeatures:
    """Extract QUIC protocol-specific features"""
    
    @staticmethod
    def extract_features(flow: Dict) -> Dict:
        """Extract QUIC features from flow"""
        features = {}
        
        packets = flow.get("packets", [])
        protocol = flow.get("protocol")
        
        # QUIC typically uses UDP
        if protocol != "UDP":
            return {"quic_score": 0, "quic_packet_sizes_variance": 0.0}
        
        # Check for QUIC-like patterns on common QUIC port (443)
        dst_port = flow.get("dst_port", 0)
        is_quic_port = dst_port in [443, 80, 4433]
        
        sizes = [p["length"] for p in packets]
        
        if len(sizes) < 10:
            return {"quic_score": 0, "quic_packet_sizes_variance": 0.0}
        
        # QUIC characteristics
        # 1. Small initial packet sizes (Initial Token)
        small_packets = sum(1 for s in sizes if s < 100)
        features["quic_small_packet_ratio"] = small_packets / len(sizes)
        
        # 2. Low variance in packet sizes (typical for encrypted QUIC)
        features["quic_packet_sizes_variance"] = float(np.var(sizes))
        features["quic_packet_sizes_std"] = float(np.std(sizes))
        features["quic_packet_sizes_cv"] = float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0
        
        # 3. Consistent packet size pattern (encryption padding)
        features["quic_size_consistency"] = QUICFeatures._calc_size_consistency(sizes)
        
        # 4. Connection ID randomness (if extractable)
        features["quic_connection_id_entropy"] = QUICFeatures._calc_cid_entropy(packets)
        
        # 5. QUIC scoring (likelihood of being QUIC)
        score = QUICFeatures._calc_quic_likelihood(flow)
        features["quic_score"] = score
        
        return features
    
    @staticmethod
    def _calc_size_consistency(sizes: list) -> float:
        """
        Measure consistency of packet sizes
        High consistency = encrypted traffic (typical for QUIC)
        """
        if len(sizes) < 2:
            return 0.0
        
        # Count packets near the median size
        median_size = np.median(sizes)
        tolerance = 50  # bytes
        
        near_median = sum(1 for s in sizes if abs(s - median_size) < tolerance)
        consistency = near_median / len(sizes)
        
        return min(consistency, 1.0)
    
    @staticmethod
    def _calc_cid_entropy(packets: list) -> float:
        """
        Estimate Connection ID entropy
        QUIC uses random connection IDs
        """
        # This is approximate without full QUIC parsing
        # We'll use packet payload entropy as proxy
        
        entropies = [p.get("payload_entropy", 0) for p in packets if p.get("payload_entropy")]
        
        if not entropies:
            return 0.0
        
        # High entropy = likely encrypted (including QUIC)
        mean_entropy = np.mean(entropies)
        
        # Entropy range is typically 0-8, normalize to 0-1
        normalized = mean_entropy / 8.0
        return min(normalized, 1.0)
    
    @staticmethod
    def _calc_quic_likelihood(flow: Dict) -> float:
        """
        Calculate likelihood that flow is QUIC
        Combines multiple indicators
        """
        score = 0.0
        
        # UDP protocol
        if flow.get("protocol") == "UDP":
            score += 0.3
        
        # Port 443 (HTTPS equivalent)
        if flow.get("dst_port") == 443:
            score += 0.2
        
        # Packet size consistency (encryption padding)
        sizes = [p["length"] for p in flow.get("packets", [])]
        if len(sizes) > 5:
            consistency = QUICFeatures._calc_size_consistency(sizes)
            score += consistency * 0.25
        
        # High payload entropy
        entropies = [p.get("payload_entropy", 0) for p in flow.get("packets", [])]
        if entropies and np.mean(entropies) > 4.5:
            score += 0.25
        
        return min(score, 1.0)


# Legacy compatibility
def quic_features(flow: Dict) -> Dict:
    """Convenience function for backward compatibility"""
    return QUICFeatures.extract_features(flow)
