"""
Advanced Flow Builder - Bidirectional flow aggregation with connection state tracking
"""
import logging
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FlowBuilder:
    """Build bidirectional flows from packets with state tracking and statistics"""
    
    def __init__(self):
        self.flows = defaultdict(list)
        self.flow_stats = {}
    
    def build_flows(self, pkts: List[Dict]) -> List[Dict]:
        """
        Build bidirectional flows from packets
        
        Args:
            pkts: List of packet dictionaries
            
        Returns:
            List of flow objects with statistics
        """
        self.flows = defaultdict(list)
        
        # Sort packets by timestamp
        pkts = sorted(pkts, key=lambda x: x["timestamp"])
        
        # Build bidirectional flows (5-tuple + reverse 5-tuple)
        for p in pkts:
            if not p.get("protocol"):
                continue
            
            # Create flow keys for both directions
            flow_key_fwd = (p["src_ip"], p["dst_ip"], p["src_port"], p["dst_port"], p["protocol"])
            flow_key_rev = (p["dst_ip"], p["src_ip"], p["dst_port"], p["src_port"], p["protocol"])
            
            # Store in forward direction (canonical)
            if p["src_ip"] < p["dst_ip"] or (p["src_ip"] == p["dst_ip"] and p["src_port"] <= p["dst_port"]):
                self.flows[flow_key_fwd].append(p)
            else:
                self.flows[flow_key_rev].append(p)
        
        # Build flow objects with statistics
        out = []
        for flow_id, packets in self.flows.items():
            if len(packets) > 0:
                flow_obj = self._create_flow_object(flow_id, packets)
                out.append(flow_obj)
        
        return out
    
    def _create_flow_object(self, flow_id: Tuple, packets: List[Dict]) -> Dict:
        """Create a flow object with comprehensive statistics"""
        
        packets = sorted(packets, key=lambda x: x["timestamp"])
        
        # Unpack flow ID
        src_ip, dst_ip, src_port, dst_port, protocol = flow_id
        
        # Basic flow info
        flow = {
            "flow_id": flow_id,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "packets": packets,
            "packet_count": len(packets),
            
            # Temporal metrics
            "start_time": packets[0]["timestamp"],
            "end_time": packets[-1]["timestamp"],
            "duration": packets[-1]["timestamp"] - packets[0]["timestamp"],
            
            # Volume metrics
            "total_bytes": sum(p["length"] for p in packets),
            "packet_sizes": [p["length"] for p in packets],
            
            # DNS metrics
            "dns_queries": [p["dns"] for p in packets if p.get("dns")],
            
            # TLS metrics
            "tls_snis": [p["tls_sni"] for p in packets if p.get("tls_sni")],
            
            # Geo metrics
            "src_geo": packets[0].get("src_geo"),
            "dst_geo": packets[0].get("dst_geo"),
        }
        
        # Calculate statistical features
        flow.update(self._calc_statistical_features(packets))
        
        # Calculate timing features
        flow.update(self._calc_timing_features(packets))
        
        # Calculate TCP-specific metrics
        if protocol == "TCP":
            flow.update(self._calc_tcp_features(packets))
        
        # Calculate entropy and randomness
        flow.update(self._calc_entropy_features(packets))
        
        return flow
    
    def _calc_statistical_features(self, packets: List[Dict]) -> Dict:
        """Calculate statistical features for flow"""
        sizes = [p["length"] for p in packets]
        
        if not sizes:
            return {}
        
        features = {
            "packet_size_mean": float(np.mean(sizes)),
            "packet_size_std": float(np.std(sizes)),
            "packet_size_min": float(np.min(sizes)),
            "packet_size_max": float(np.max(sizes)),
            "packet_size_median": float(np.median(sizes)),
            "packet_size_skewness": float(self._skewness(sizes)),
            "packet_size_kurtosis": float(self._kurtosis(sizes)),
            
            # Directionality
            "fwd_packet_count": sum(1 for p in packets if not self._is_reverse(p, packets[0])),
            "bwd_packet_count": sum(1 for p in packets if self._is_reverse(p, packets[0])),
            
            # Byte distribution
            "total_bytes_fwd": sum(p["length"] for p in packets if not self._is_reverse(p, packets[0])),
            "total_bytes_bwd": sum(p["length"] for p in packets if self._is_reverse(p, packets[0])),
        }
        
        # Add asymmetry metric
        if features["fwd_packet_count"] > 0:
            features["upload_download_ratio"] = features["total_bytes_fwd"] / (features["total_bytes_bwd"] + 1)
        
        return features
    
    def _calc_timing_features(self, packets: List[Dict]) -> Dict:
        """Calculate inter-packet timing features"""
        timestamps = [p["timestamp"] for p in packets]
        
        if len(timestamps) < 2:
            return {"mean_iat": 0.0, "std_iat": 0.0, "min_iat": 0.0, "max_iat": 0.0}
        
        # Inter-arrival times
        iats = np.diff(timestamps)
        
        features = {
            "mean_iat": float(np.mean(iats)),
            "std_iat": float(np.std(iats)),
            "min_iat": float(np.min(iats)),
            "max_iat": float(np.max(iats)),
            "median_iat": float(np.median(iats)),
            
            # Burstiness (Hurst parameter approximation)
            "burstiness": float(np.std(iats) / (np.mean(iats) + 1e-6)),
        }
        
        return features
    
    def _calc_tcp_features(self, packets: List[Dict]) -> Dict:
        """Calculate TCP-specific features"""
        tcp_packets = [p for p in packets if p.get("flags")]
        
        if not tcp_packets:
            return {}
        
        features = {
            "syn_count": sum(1 for p in tcp_packets if "S" in str(p.get("flags", ""))),
            "fin_count": sum(1 for p in tcp_packets if "F" in str(p.get("flags", ""))),
            "rst_count": sum(1 for p in tcp_packets if "R" in str(p.get("flags", ""))),
            "ack_count": sum(1 for p in tcp_packets if "A" in str(p.get("flags", ""))),
            "psh_count": sum(1 for p in tcp_packets if "P" in str(p.get("flags", ""))),
        }
        
        return features
    
    def _calc_entropy_features(self, packets: List[Dict]) -> Dict:
        """Calculate entropy and randomness features"""
        features = {
            "payload_entropy_mean": float(np.mean([p.get("payload_entropy", 0) for p in packets])),
            "payload_entropy_std": float(np.std([p.get("payload_entropy", 0) for p in packets])),
        }
        
        return features
    
    def _is_reverse(self, pkt: Dict, first_pkt: Dict) -> bool:
        """Check if packet is in reverse direction"""
        return pkt["src_ip"] == first_pkt["dst_ip"]
    
    def _skewness(self, data):
        """Calculate skewness of distribution"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((np.array(data) - mean) / std) ** 3)
    
    def _kurtosis(self, data):
        """Calculate kurtosis of distribution"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((np.array(data) - mean) / std) ** 4) - 3


def build_flows(pkts: List[Dict]) -> List[Dict]:
    """Convenience function for backward compatibility"""
    builder = FlowBuilder()
    return builder.build_flows(pkts)
