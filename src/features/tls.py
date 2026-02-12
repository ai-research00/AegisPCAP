"""
TLS/HTTPS Features - JA3 fingerprinting and TLS-based anomaly detection
"""
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class TLSFeatures:
    """Extract TLS-based anomaly detection features"""
    
    # Common TLS versions
    TLS_VERSIONS = {
        0x0303: "TLS1.2",
        0x0304: "TLS1.3",
        0x0302: "TLS1.1",
        0x0301: "TLS1.0",
    }
    
    @staticmethod
    def extract_features(flow: Dict) -> Dict:
        """Extract TLS features from flow"""
        features = {}
        
        tls_snis = flow.get("tls_snis", [])
        packets = flow.get("packets", [])
        tls_packets = [p for p in packets if p.get("tls_sni") or p.get("tls_version")]
        
        if not tls_packets:
            return {
                "tls_sni_count": 0,
                "tls_unique_snis": 0,
                "tls_entropy": 0.0,
                "tls_self_signed_score": 0.0,
                "tls_reuse_score": 0.0,
                "tls_c2_score": 0.0,
            }
        
        # SNI-based features
        snis = [p.get("tls_sni") for p in tls_packets if p.get("tls_sni")]
        features["tls_sni_count"] = len(snis)
        features["tls_unique_snis"] = len(set(snis))
        features["tls_sni_uniqueness"] = len(set(snis)) / max(len(snis), 1)
        
        # SNI entropy
        if snis:
            from src.features.dns import entropy
            sni_entropies = [entropy(s) for s in snis]
            features["tls_entropy"] = float(np.mean(sni_entropies)) if sni_entropies else 0.0
            features["tls_entropy_max"] = float(max(sni_entropies)) if sni_entropies else 0.0
        
        # Certificate-based features (if extracted)
        features["tls_certificate_diversity"] = TLSFeatures._calc_cert_diversity(tls_packets)
        features["tls_self_signed_score"] = TLSFeatures._calc_self_signed_score(snis)
        
        # Session reuse detection
        features["tls_reuse_score"] = TLSFeatures._calc_reuse_score(tls_packets)
        
        # C2 detection
        features["tls_c2_score"] = TLSFeatures._calc_c2_score(tls_packets, snis)
        
        # Version analysis
        tls_versions = [p.get("tls_version") for p in tls_packets if p.get("tls_version")]
        if tls_versions:
            outdated_count = sum(1 for v in tls_versions if v and v < 0x0303)  # TLS 1.2 = 0x0303
            features["tls_outdated_ratio"] = outdated_count / len(tls_versions)
        
        return features
    
    @staticmethod
    def _calc_cert_diversity(tls_packets: List[Dict]) -> float:
        """Calculate certificate diversity (indicates rotating certificates for C2)"""
        snis = [p.get("tls_sni") for p in tls_packets if p.get("tls_sni")]
        
        if not snis:
            return 0.0
        
        # High SNI diversity = many different certificates/domains
        unique_snis = len(set(snis))
        diversity = unique_snis / len(snis)
        
        return min(diversity, 1.0)
    
    @staticmethod
    def _calc_self_signed_score(snis: List[str]) -> float:
        """
        Detect self-signed certificate indicators
        - Unusual CN/SAN patterns
        - Localhost/internal hostnames
        - IP addresses in SNI
        """
        if not snis:
            return 0.0
        
        score = 0.0
        
        for sni in snis:
            # IP address in SNI (unusual)
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", sni):
                score += 0.3
            
            # Localhost/internal
            if sni in ["localhost", "127.0.0.1"] or sni.endswith(".local"):
                score += 0.3
            
            # Generic self-signed patterns
            if any(pattern in sni.lower() for pattern in ["self-signed", "generated", "cert"]):
                score += 0.2
        
        return min(score, 1.0)
    
    @staticmethod
    def _calc_reuse_score(tls_packets: List[Dict]) -> float:
        """
        Detect TLS session reuse (multiple connections with same SessionID/tickets)
        Indicators of beaconing or C2
        """
        if len(tls_packets) < 2:
            return 0.0
        
        # Session ID reuse (if available)
        # Note: This would require extracting session IDs from TLS handshakes
        # For now, approximate based on repeated SNIs with same version
        
        snis = [p.get("tls_sni") for p in tls_packets if p.get("tls_sni")]
        tls_versions = [p.get("tls_version") for p in tls_packets if p.get("tls_version")]
        
        if not snis or len(snis) < 2:
            return 0.0
        
        # Count repeated SNI+version combinations
        pairs = list(zip(snis, tls_versions))
        unique_pairs = len(set(pairs))
        reuse_ratio = 1 - (unique_pairs / len(pairs))
        
        return min(reuse_ratio, 1.0)
    
    @staticmethod
    def _calc_c2_score(tls_packets: List[Dict], snis: List[str]) -> float:
        """
        Detect TLS-based C2 indicators
        - Session reuse
        - Cert rotation
        - Unusual TLS extensions
        - Non-matching hostname/certificate
        """
        if not snis:
            return 0.0
        
        score = 0.0
        
        # Session reuse (high score)
        reuse_score = TLSFeatures._calc_reuse_score(tls_packets)
        score += reuse_score * 0.3
        
        # Certificate diversity (rotating certs)
        diversity = TLSFeatures._calc_cert_diversity(tls_packets)
        if diversity > 0.7:
            score += 0.3
        
        # Unusual SNI patterns
        for sni in snis:
            # Very long SNI (data exfiltration)
            if len(sni) > 100:
                score += 0.2
            
            # Numeric SNI (generated)
            if sni.replace(".", "").isdigit():
                score += 0.2
        
        # Check for GREASE/unusual extensions (advanced evasion)
        # Would require parsing ClientHello details
        
        return min(score, 1.0)
    
    @staticmethod
    def ja3_string(ssl_version: int, ciphers: List[int], extensions: List[int],
                   elliptic_curves: List[int], ec_points: List[int]) -> Tuple[str, str]:
        """
        Create JA3 fingerprint string and hash
        
        JA3 format: SSLVersion,Ciphers,Extensions,EllipticCurves,EllipticCurveFormats
        """
        ja3_str = f"{ssl_version},{'-'.join(map(str, ciphers))},{'-'.join(map(str, extensions))},{'-'.join(map(str, elliptic_curves))},{'-'.join(map(str, ec_points))}"
        ja3_hash = hashlib.md5(ja3_str.encode()).hexdigest()
        
        return ja3_str, ja3_hash


# Legacy compatibility
def ja3_string(v, c, e, ec, pf):
    """Convenience function for backward compatibility"""
    return TLSFeatures.ja3_string(v, c, e, ec, pf)


import numpy as np

