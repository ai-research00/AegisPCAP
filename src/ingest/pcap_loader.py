"""
Advanced PCAP Loading Module - Multi-format support with data quality checks
"""
import logging
from typing import List, Dict, Optional
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, IPv6, Raw
from scapy.layers.dns import DNS

# Try to import TLS, but make it optional if not available
try:
    from scapy.layers.tls import TLS
except (ImportError, ModuleNotFoundError):
    TLS = None
    
import geoip2.database
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

class PcapLoader:
    """Enhanced PCAP loader with quality metrics, protocol support, and GeoIP"""
    
    def __init__(self, geoip_db_path: Optional[str] = None):
        self.geoip_db = None
        self.quality_metrics = {}
        if geoip_db_path:
            try:
                self.geoip_db = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning(f"GeoIP database not available: {e}")
    
    def load_pcap(self, path: str, validate: bool = True) -> List[Dict]:
        """
        Load PCAP with comprehensive packet parsing and quality checks
        
        Args:
            path: Path to PCAP file
            validate: Enable data quality validation
            
        Returns:
            List of parsed packet records
        """
        try:
            packets = rdpcap(path)
        except Exception as e:
            logger.error(f"Failed to load PCAP: {e}")
            return []
        
        out = []
        dropped = 0
        
        for p in packets:
            try:
                rec = self._parse_packet(p)
                if rec:
                    if validate:
                        if self._validate_packet(rec):
                            out.append(rec)
                        else:
                            dropped += 1
                    else:
                        out.append(rec)
            except Exception as e:
                logger.debug(f"Packet parse error: {e}")
                dropped += 1
        
        self.quality_metrics = {
            "total_packets": len(packets),
            "valid_packets": len(out),
            "dropped_packets": dropped,
            "drop_rate": dropped / len(packets) if packets else 0
        }
        
        return out
    
    def _parse_packet(self, p) -> Optional[Dict]:
        """Parse single packet with L2-L7 extraction"""
        
        # Skip non-IP packets
        if IP not in p and IPv6 not in p:
            return None
        
        ip_layer = p[IP] if IP in p else p[IPv6]
        
        rec = {
            "timestamp": float(p.time),
            "src_ip": ip_layer.src,
            "dst_ip": ip_layer.dst,
            "length": len(p),
            "ip_version": 4 if IP in p else 6,
            "ttl": ip_layer.ttl if hasattr(ip_layer, 'ttl') else None,
            "protocol": None,
            "src_port": None,
            "dst_port": None,
            "flags": None,
            "seq": None,
            "ack": None,
            "dns": None,
            "tls_sni": None,
            "tls_version": None,
            "http_method": None,
            "http_host": None,
            "payload_entropy": self._calc_entropy(bytes(p)),
            "packet_hash": hashlib.md5(bytes(p)).hexdigest()
        }
        
        # TCP/UDP parsing
        if TCP in p:
            rec["protocol"] = "TCP"
            rec["src_port"] = p[TCP].sport
            rec["dst_port"] = p[TCP].dport
            rec["flags"] = p[TCP].flags
            rec["seq"] = p[TCP].seq
            rec["ack"] = p[TCP].ack
        elif UDP in p:
            rec["protocol"] = "UDP"
            rec["src_port"] = p[UDP].sport
            rec["dst_port"] = p[UDP].dport
        elif ICMP in p:
            rec["protocol"] = "ICMP"
            rec["type"] = p[ICMP].type
            rec["code"] = p[ICMP].code
        else:
            return None
        
        # DNS extraction
        if DNS in p and p[DNS].qd:
            try:
                rec["dns"] = p[DNS].qd.qname.decode(errors="ignore")
            except:
                pass
        
        # TLS extraction (basic SNI)
        if TLS in p:
            try:
                rec["tls_version"] = p[TLS].version
                if hasattr(p[TLS], 'server_name'):
                    rec["tls_sni"] = p[TLS].server_name
            except:
                pass
        
        # GeoIP lookup
        rec["src_geo"] = self._geoip_lookup(rec["src_ip"])
        rec["dst_geo"] = self._geoip_lookup(rec["dst_ip"])
        
        return rec
    
    def _validate_packet(self, rec: Dict) -> bool:
        """Validate packet for data quality"""
        # Ensure required fields present
        if not rec.get("protocol"):
            return False
        
        # Check timestamp validity
        if rec["timestamp"] < 0 or rec["timestamp"] > 2e9:
            return False
        
        # Check port validity
        if rec["src_port"] is not None and (rec["src_port"] < 0 or rec["src_port"] > 65535):
            return False
        if rec["dst_port"] is not None and (rec["dst_port"] < 0 or rec["dst_port"] > 65535):
            return False
        
        return True
    
    def _geoip_lookup(self, ip: str) -> Optional[Dict]:
        """GeoIP lookup for IP address"""
        if not self.geoip_db:
            return None
        try:
            response = self.geoip_db.city(ip)
            return {
                "country": response.country.iso_code,
                "city": response.city.name,
                "latitude": response.location.latitude,
                "longitude": response.location.longitude
            }
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip}: {e}")
            return None
    
    def _calc_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of payload"""
        if not data:
            return 0.0
        
        counts = Counter(data)
        entropy = 0.0
        for count in counts.values():
            p = count / len(data)
            entropy -= p * (p.bit_length() - 1 if p > 0 else 0)
        return entropy
    
    def get_quality_metrics(self) -> Dict:
        """Get data quality metrics from last load"""
        return self.quality_metrics


def load_pcap(path: str) -> List[Dict]:
    """Convenience function for backward compatibility"""
    loader = PcapLoader()
    return loader.load_pcap(path)
