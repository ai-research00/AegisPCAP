"""
DNS Features - Advanced DNS heuristics for C2, DGA, and tunnel detection
"""
import math
import logging
from collections import Counter
from typing import Dict, List
import re

logger = logging.getLogger(__name__)


def entropy(s: str) -> float:
    """Calculate Shannon entropy of string"""
    if not s:
        return 0.0
    c = Counter(s)
    t = len(s)
    return -sum((v / t) * math.log2(v / t) for v in c.values() if v > 0)


class DNSFeatures:
    """Extract DNS-based anomaly detection features"""
    
    @staticmethod
    def extract_features(flow: Dict) -> Dict:
        """Extract DNS features from flow"""
        features = {}
        
        dns_queries = flow.get("dns_queries", [])
        
        if not dns_queries:
            return {
                "dns_entropy": 0.0,
                "dns_query_count": 0,
                "dns_unique_domains": 0,
                "dns_avg_label_len": 0.0,
                "dns_subdomain_count": 0.0,
                "dns_numeric_ratio": 0.0,
                "dns_entropy_variance": 0.0,
                "dns_beaconing_score": 0.0,
            }
        
        # Basic entropy
        entropies = [entropy(q) for q in dns_queries]
        features["dns_entropy"] = float(np.mean(entropies)) if entropies else 0.0
        features["dns_entropy_max"] = float(np.max(entropies)) if entropies else 0.0
        features["dns_entropy_min"] = float(np.min(entropies)) if entropies else 0.0
        features["dns_entropy_variance"] = float(np.var(entropies)) if len(entropies) > 1 else 0.0
        
        # Query patterns
        features["dns_query_count"] = len(dns_queries)
        features["dns_unique_domains"] = len(set(dns_queries))
        features["dns_query_uniqueness"] = features["dns_unique_domains"] / max(features["dns_query_count"], 1)
        
        # Domain characteristics
        features["dns_avg_label_len"] = float(np.mean([len(label) for q in dns_queries for label in q.split(".")]))
        features["dns_avg_domain_len"] = float(np.mean([len(q) for q in dns_queries]))
        
        # Subdomain enumeration detection
        domain_prefixes = {}
        for q in dns_queries:
            labels = q.split(".")
            if len(labels) > 2:
                domain = ".".join(labels[-2:])
                domain_prefixes[domain] = domain_prefixes.get(domain, 0) + 1
        
        if domain_prefixes:
            max_subdomains = max(domain_prefixes.values())
            features["dns_subdomain_count"] = float(max_subdomains)
            features["dns_subdomain_entropy"] = entropy("".join([str(v) for v in domain_prefixes.values()]))
        else:
            features["dns_subdomain_count"] = 0.0
            features["dns_subdomain_entropy"] = 0.0
        
        # Numeric/random character ratio (DGA indicator)
        numeric_count = sum(1 for q in dns_queries for c in q if c.isdigit())
        total_chars = sum(len(q) for q in dns_queries)
        features["dns_numeric_ratio"] = numeric_count / max(total_chars, 1)
        
        # DGA detection
        features["dns_dga_score"] = DNSFeatures._calc_dga_score(dns_queries)
        
        # Beaconing detection
        features["dns_beaconing_score"] = DNSFeatures._calc_beaconing_score(flow)
        
        # Tunneling detection
        features["dns_tunnel_score"] = DNSFeatures._calc_tunnel_score(dns_queries)
        
        return features
    
    @staticmethod
    def _calc_dga_score(domains: List[str]) -> float:
        """
        Calculate DGA likelihood score
        DGA domains typically have:
        - High entropy
        - Long subdomains
        - Numeric characters
        - Infrequent patterns
        """
        if not domains:
            return 0.0
        
        score = 0.0
        domain_entropy_score = np.mean([entropy(d) for d in domains]) / 8.0  # Normalize to 0-1
        score += domain_entropy_score * 0.4
        
        # Check for numeric dominance
        numeric_ratio = sum(1 for d in domains for c in d if c.isdigit()) / sum(len(d) for d in domains)
        score += min(numeric_ratio * 2, 1.0) * 0.2
        
        # Check subdomain length variance
        subdomain_lens = [len(d.split(".")[0]) for d in domains]
        if len(subdomain_lens) > 1:
            variance = np.var(subdomain_lens)
            score += min(variance / 100, 1.0) * 0.2
        
        # Uniqueness (new domains frequently)
        uniqueness = len(set(domains)) / len(domains)
        score += uniqueness * 0.2
        
        return min(score, 1.0)
    
    @staticmethod
    def _calc_beaconing_score(flow: Dict) -> float:
        """
        Detect periodic beaconing behavior
        - Regular DNS query intervals
        - Repeating domain patterns
        """
        packets = flow.get("packets", [])
        dns_packets = [p for p in packets if p.get("dns")]
        
        if len(dns_packets) < 3:
            return 0.0
        
        # Check timing regularity
        timestamps = [p["timestamp"] for p in dns_packets]
        iats = np.diff(timestamps)
        
        if len(iats) > 1:
            iat_cv = np.std(iats) / np.mean(iats)
            regularity_score = max(0, 1 - iat_cv)  # Low CV = high regularity
        else:
            regularity_score = 0.0
        
        # Check domain repetition
        domains = [p["dns"] for p in dns_packets]
        unique_ratio = len(set(domains)) / len(domains)
        repetition_score = 1 - unique_ratio
        
        # Combine scores
        beaconing_score = (regularity_score * 0.6 + repetition_score * 0.4)
        
        return min(beaconing_score, 1.0)
    
    @staticmethod
    def _calc_tunnel_score(domains: List[str]) -> float:
        """
        Detect DNS tunneling attempts
        - Unusually long subdomains
        - Base64/hex encoded data
        - Many queries with data-like patterns
        """
        if not domains:
            return 0.0
        
        score = 0.0
        
        # Check for long subdomains (typical tunneling)
        avg_subdomain_len = np.mean([len(d.split(".")[0]) for d in domains])
        if avg_subdomain_len > 20:
            score += 0.4
        
        # Check for base64-like patterns (A-Z, a-z, 0-9, -, _)
        b64_pattern = re.compile(r"[A-Za-z0-9_-]{15,}")
        b64_matches = sum(1 for d in domains if b64_pattern.search(d))
        score += min((b64_matches / len(domains)) * 0.6, 0.3)
        
        # High entropy in subdomains
        subdomain_entropies = [entropy(d.split(".")[0]) for d in domains]
        avg_entropy = np.mean(subdomain_entropies)
        if avg_entropy > 4.5:
            score += 0.3
        
        return min(score, 1.0)


# Legacy compatibility
def dns_features(flow: Dict) -> Dict:
    """Convenience function for backward compatibility"""
    return DNSFeatures.extract_features(flow)


import numpy as np

