"""
Advanced Agent Reasoning Engine - Multi-source evidence correlation and MITRE ATT&CK mapping
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EvidenceCorrelator:
    """Correlate multiple evidence sources for threat analysis"""
    
    # MITRE ATT&CK Framework Tactics
    MITRE_TACTICS = {
        'reconnaissance': ['Discovery', 'Passive discovery'],
        'resource_development': ['Domain registration', 'Infrastructure setup'],
        'initial_access': ['Phishing', 'External remote services'],
        'execution': ['Command execution', 'Scripting'],
        'persistence': ['Account creation', 'Persistence mechanisms'],
        'privilege_escalation': ['Elevation', 'Privilege abuse'],
        'defense_evasion': ['Encryption', 'Masquerading', 'Obfuscation'],
        'credential_access': ['Credential dumping', 'Brute force'],
        'discovery': ['Internal reconnaissance', 'Network service discovery'],
        'lateral_movement': ['Remote service exploitation', 'Internal propagation'],
        'collection': ['Data staging', 'Data exfiltration prep'],
        'command_and_control': ['C2 communication', 'Dead drop resolver'],
        'exfiltration': ['Data transfer', 'Exfiltration over C2'],
        'impact': ['Data destruction', 'Service disruption']
    }
    
    # MITRE ATT&CK Techniques to Features Mapping
    TECHNIQUE_FEATURES = {
        'C2_communication': ['dns_beaconing_score', 'tls_c2_score', 'mean_iat', 'std_iat', 'timing_periodicity'],
        'data_exfiltration': ['total_bytes_bwd', 'byte_rate', 'upload_download_ratio', 'dns_tunnel_score'],
        'dns_abuse': ['dns_entropy', 'dns_dga_score', 'dns_tunnel_score', 'dns_query_uniqueness'],
        'beaconing': ['mean_iat', 'timing_periodicity', 'packet_rate', 'beaconing_score'],
        'lateral_movement': ['dns_unique_domains', 'port_scan_indicators', 'failed_auth_attempts'],
        'tls_evasion': ['tls_entropy', 'tls_self_signed_score', 'tls_certificate_diversity'],
        'port_scanning': ['syn_count', 'rst_count', 'packet_rate'],
        'credential_abuse': ['failed_attempts', 'authentication_spike'],
    }
    
    def __init__(self):
        self.evidence_weights = {
            'anomaly_score': 0.25,
            'dns_indicators': 0.20,
            'tls_indicators': 0.20,
            'timing_indicators': 0.15,
            'behavioral_score': 0.20
        }
    
    def correlate_evidence(self, flow: Dict, ml_scores: Dict, heuristic_scores: Dict) -> Dict:
        """
        Correlate multiple evidence sources
        
        Args:
            flow: Flow dictionary
            ml_scores: Machine learning model scores
            heuristic_scores: Heuristic-based scores
            
        Returns:
            Correlated verdict with evidence trail
        """
        
        verdict = {
            "risk_score": 0.0,
            "confidence": 0.0,
            "threat_types": [],
            "mitre_tactics": [],
            "evidence": [],
            "false_positive_likelihood": 0.0,
            "requires_investigation": False,
            "recommended_action": None
        }
        
        # Aggregate ML scores
        ml_risk = np.mean([v for v in ml_scores.values() if isinstance(v, (int, float))]) if ml_scores else 0
        
        # Aggregate heuristic scores
        heuristic_risk = self._aggregate_heuristics(flow, heuristic_scores)
        
        # Detect threat types
        threats = self._detect_threat_types(flow, ml_scores, heuristic_scores)
        verdict["threat_types"] = threats
        
        # Map to MITRE ATT&CK
        mitre_tactics = self._map_to_mitre(threats, flow)
        verdict["mitre_tactics"] = mitre_tactics
        
        # Combined risk scoring
        verdict["risk_score"] = self._calc_combined_risk(ml_risk, heuristic_risk, flow)
        
        # Confidence assessment
        verdict["confidence"] = self._calc_confidence(flow, ml_scores, threats)
        
        # Evidence summary
        verdict["evidence"] = self._summarize_evidence(flow, ml_scores, heuristic_scores, threats)
        
        # False positive likelihood
        verdict["false_positive_likelihood"] = self._estimate_fp_likelihood(flow, verdict["risk_score"])
        
        # Decision
        if verdict["risk_score"] > 0.7 and verdict["confidence"] > 0.6:
            verdict["requires_investigation"] = True
            verdict["recommended_action"] = "isolate_and_investigate"
        elif verdict["risk_score"] > 0.5:
            verdict["requires_investigation"] = True
            verdict["recommended_action"] = "hunt"
        elif verdict["risk_score"] > 0.3:
            verdict["recommended_action"] = "monitor"
        else:
            verdict["recommended_action"] = "allow"
        
        return verdict
    
    def _aggregate_heuristics(self, flow: Dict, heuristic_scores: Dict) -> float:
        """Aggregate heuristic scores with evidence weighting"""
        
        scores = []
        
        # DNS indicators
        dns_score = max(
            flow.get("dns_entropy", 0) / 8.0,  # Normalize entropy
            heuristic_scores.get("dns_beaconing_score", 0),
            heuristic_scores.get("dns_dga_score", 0),
            heuristic_scores.get("dns_tunnel_score", 0)
        )
        scores.append(("dns_indicators", dns_score, 0.20))
        
        # TLS indicators
        tls_score = max(
            heuristic_scores.get("tls_c2_score", 0),
            heuristic_scores.get("tls_self_signed_score", 0),
            heuristic_scores.get("tls_reuse_score", 0)
        )
        scores.append(("tls_indicators", tls_score, 0.20))
        
        # Timing indicators
        timing_score = max(
            heuristic_scores.get("timing_periodicity", 0),
            heuristic_scores.get("beaconing_score", 0),
            heuristic_scores.get("packet_arrival_regularity", 0)
        )
        scores.append(("timing_indicators", timing_score, 0.15))
        
        # Behavioral score
        behavioral_score = (
            heuristic_scores.get("activity_burstiness", 0) * 0.3 +
            max(0, (flow.get("upload_download_ratio", 1) - 0.5) / 1.5) * 0.7  # Asymmetric traffic
        )
        scores.append(("behavioral", behavioral_score, 0.15))
        
        # Weighted aggregation
        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0
        
        return min(weighted_score, 1.0)
    
    def _detect_threat_types(self, flow: Dict, ml_scores: Dict, heuristic_scores: Dict) -> List[str]:
        """Detect specific threat types from evidence"""
        
        threats = []
        
        # C2 Detection
        c2_score = ml_scores.get("c2_score", 0)
        tls_c2 = heuristic_scores.get("tls_c2_score", 0)
        dns_beacon = heuristic_scores.get("dns_beaconing_score", 0)
        
        if (c2_score > 0.6 or (tls_c2 > 0.5 and dns_beacon > 0.5)):
            threats.append("C2_communication")
        
        # Data Exfiltration
        exfil_score = ml_scores.get("exfil_score", 0)
        upload_ratio = flow.get("upload_download_ratio", 0)
        
        if (exfil_score > 0.6 or (upload_ratio > 2.0 and flow.get("total_bytes_bwd", 0) > 1000000)):
            threats.append("data_exfiltration")
        
        # DNS Abuse
        dns_entropy = flow.get("dns_entropy", 0) / 8.0
        dga_score = heuristic_scores.get("dns_dga_score", 0)
        
        if dns_entropy > 0.7 or dga_score > 0.6:
            threats.append("dns_abuse")
        
        # Beaconing
        beacon_score = heuristic_scores.get("beaconing_score", 0)
        botnet_score = ml_scores.get("botnet_score", 0)
        
        if beacon_score > 0.7 or botnet_score > 0.6:
            threats.append("beaconing")
        
        # Port Scanning
        if flow.get("syn_count", 0) > 100 and flow.get("packet_count", 0) > 200:
            threats.append("port_scanning")
        
        # TLS Evasion
        tls_self_signed = heuristic_scores.get("tls_self_signed_score", 0)
        tls_cert_div = heuristic_scores.get("tls_certificate_diversity", 0)
        
        if tls_self_signed > 0.5 or tls_cert_div > 0.6:
            threats.append("tls_evasion")
        
        return threats
    
    def _map_to_mitre(self, threats: List[str], flow: Dict) -> List[str]:
        """Map detected threats to MITRE ATT&CK tactics"""
        
        tactics = []
        
        for threat in threats:
            if threat == "C2_communication":
                tactics.extend(["command_and_control", "defense_evasion"])
            elif threat == "data_exfiltration":
                tactics.extend(["exfiltration", "collection"])
            elif threat == "dns_abuse":
                tactics.extend(["command_and_control", "defense_evasion"])
            elif threat == "beaconing":
                tactics.append("command_and_control")
            elif threat == "port_scanning":
                tactics.extend(["reconnaissance", "discovery"])
            elif threat == "lateral_movement":
                tactics.extend(["lateral_movement", "discovery"])
            elif threat == "tls_evasion":
                tactics.extend(["defense_evasion", "command_and_control"])
        
        return list(set(tactics))
    
    def _calc_combined_risk(self, ml_risk: float, heuristic_risk: float, flow: Dict) -> float:
        """Calculate combined risk score with contextual adjustment"""
        
        # Base combination
        combined = ml_risk * 0.5 + heuristic_risk * 0.5
        
        # Context adjustment
        # Adjust based on flow characteristics
        if flow.get("duration", 0) < 1:
            combined *= 0.8  # Lower confidence for very short flows
        
        if flow.get("packet_count", 0) < 5:
            combined *= 0.7  # Lower confidence for few packets
        
        return min(combined, 1.0)
    
    def _calc_confidence(self, flow: Dict, ml_scores: Dict, threats: List[str]) -> float:
        """Calculate confidence level in the assessment"""
        
        confidence = 0.5  # Base confidence
        
        # More flows = higher confidence
        packet_count = flow.get("packet_count", 0)
        if packet_count > 100:
            confidence += 0.3
        elif packet_count > 10:
            confidence += 0.2
        
        # Multiple threat indicators = higher confidence
        confidence += min(len(threats) * 0.1, 0.3)
        
        # If multiple ML models agree = higher confidence
        ml_agreement = sum(1 for v in ml_scores.values() if isinstance(v, (int, float)) and v > 0.5)
        confidence += min(ml_agreement * 0.05, 0.2)
        
        return min(confidence, 1.0)
    
    def _summarize_evidence(self, flow: Dict, ml_scores: Dict, heuristic_scores: Dict, threats: List[str]) -> List[Dict]:
        """Summarize evidence in human-readable format"""
        
        evidence = []
        
        # ML-based evidence
        for model, score in ml_scores.items():
            if isinstance(score, (int, float)) and score > 0.5:
                evidence.append({
                    "source": model,
                    "severity": "high" if score > 0.7 else "medium",
                    "score": score,
                    "description": f"{model} detected suspicious pattern"
                })
        
        # Heuristic evidence
        if heuristic_scores.get("dns_entropy", 0) > 0.7:
            evidence.append({
                "source": "DNS_entropy",
                "severity": "medium",
                "score": heuristic_scores.get("dns_entropy", 0) / 8.0,
                "description": "High entropy in DNS queries (possible DGA)"
            })
        
        if heuristic_scores.get("beaconing_score", 0) > 0.6:
            evidence.append({
                "source": "Beaconing",
                "severity": "high",
                "score": heuristic_scores.get("beaconing_score", 0),
                "description": "Periodic communication pattern detected"
            })
        
        return evidence
    
    def _estimate_fp_likelihood(self, flow: Dict, risk_score: float) -> float:
        """Estimate false positive likelihood"""
        
        fp_likelihood = 0.5 - (risk_score * 0.3)  # High risk = low FP likelihood
        
        # Adjust based on benign indicators
        if flow.get("protocol") == "TCP" and flow.get("dst_port") == 443:
            fp_likelihood += 0.1  # HTTPS is common
        
        if flow.get("packet_count", 0) < 5:
            fp_likelihood += 0.2  # Short flows = higher FP likelihood
        
        return max(0, min(fp_likelihood, 1.0))


def reason(scores: Dict) -> Dict:
    """Legacy function for backward compatibility"""
    risk = sum(v for v in scores.values() if isinstance(v, (int, float)))
    return {"risk": min(100, risk * 30)}

