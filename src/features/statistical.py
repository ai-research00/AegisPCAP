"""
Statistical Features - Comprehensive flow statistics for anomaly detection
"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class StatisticalFeatures:
    """Extract comprehensive statistical features from flows"""
    
    @staticmethod
    def extract_features(flow: Dict) -> Dict:
        """Extract all statistical features from a flow"""
        features = {}
        
        # Already calculated in flow_builder
        for key in [
            "packet_size_mean", "packet_size_std", "packet_size_min", "packet_size_max",
            "packet_size_median", "packet_size_skewness", "packet_size_kurtosis",
            "fwd_packet_count", "bwd_packet_count", "total_bytes_fwd", "total_bytes_bwd",
            "upload_download_ratio", "mean_iat", "std_iat", "min_iat", "max_iat",
            "burstiness", "payload_entropy_mean", "payload_entropy_std"
        ]:
            if key in flow:
                features[key] = flow[key]
        
        # Additional statistical metrics
        packets = flow.get("packets", [])
        if packets:
            sizes = [p["length"] for p in packets]
            
            # Packet size distribution percentiles
            features["pkt_size_p25"] = float(np.percentile(sizes, 25)) if sizes else 0
            features["pkt_size_p50"] = float(np.percentile(sizes, 50)) if sizes else 0
            features["pkt_size_p75"] = float(np.percentile(sizes, 75)) if sizes else 0
            features["pkt_size_p95"] = float(np.percentile(sizes, 95)) if sizes else 0
            features["pkt_size_p99"] = float(np.percentile(sizes, 99)) if sizes else 0
            
            # Coefficient of variation
            if features.get("packet_size_mean", 0) > 0:
                features["pkt_size_cv"] = features.get("packet_size_std", 0) / features["packet_size_mean"]
            
            # Packet arrival rate (packets per second)
            duration = flow.get("duration", 1)
            if duration > 0:
                features["packet_rate"] = flow.get("packet_count", 0) / duration
                features["byte_rate"] = flow.get("total_bytes", 0) / duration
        
        return features
