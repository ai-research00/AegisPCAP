"""
AegisPCAP Main Pipeline - End-to-end PCAP analysis orchestration
"""
import logging
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

# Import pipeline components
from src.ingest.pcap_loader import PcapLoader
from src.ingest.flow_builder import FlowBuilder
from src.features import dns, tls, timing, statistical, quic
from src.models.ensemble import AnomalyEnsemble, C2Detector, DataExfilDetector, BotnetDetector
from src.agent.planner import TaskPlanner
from src.agent.reasoning import EvidenceCorrelator
from config import *

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """End-to-end network traffic analysis pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize pipeline with configuration"""
        
        self.config = config or {}
        
        # Initialize components
        self.pcap_loader = PcapLoader(geoip_db_path=GEOIP_DB_PATH)
        self.flow_builder = FlowBuilder()
        self.task_planner = TaskPlanner()
        self.evidence_correlator = EvidenceCorrelator()
        
        # Initialize ML models (lazy load)
        self.anomaly_ensemble = None
        self.c2_detector = None
        self.exfil_detector = None
        self.botnet_detector = None
        
        self.is_trained = False
        
        logger.info(f"Initialized {PROJECT_NAME} v{VERSION}")
    
    def analyze_pcap(self, pcap_path: str, query: Optional[str] = None) -> Dict:
        """
        Complete PCAP analysis pipeline
        
        Args:
            pcap_path: Path to PCAP file
            query: Optional user query for analysis
            
        Returns:
            Analysis results with findings and verdicts
        """
        
        logger.info(f"Starting analysis of {pcap_path}")
        
        # Phase 1: Load and parse PCAP
        logger.info("Phase 1: Loading PCAP...")
        packets = self.pcap_loader.load_pcap(pcap_path, validate=VALIDATE_PACKETS)
        
        if not packets:
            logger.error(f"Failed to load packets from {pcap_path}")
            return {"error": "No valid packets found"}
        
        logger.info(f"Loaded {len(packets)} packets, Quality metrics: {self.pcap_loader.get_quality_metrics()}")
        
        # Phase 2: Build flows
        logger.info("Phase 2: Building flows...")
        flows = self.flow_builder.build_flows(packets)
        logger.info(f"Built {len(flows)} flows")
        
        # Phase 3: Extract features
        logger.info("Phase 3: Extracting features...")
        flows_with_features = self._extract_all_features(flows)
        
        # Phase 4: Plan analysis (if query provided)
        if query:
            plan = self.task_planner.plan(query)
            logger.info(f"Analysis plan: {plan['tools']}")
        
        # Phase 5: Run ML detection
        logger.info("Phase 5: Running ML models...")
        flows_with_ml = self._run_ml_inference(flows_with_features)
        
        # Phase 6: Evidence correlation and verdict
        logger.info("Phase 6: Correlating evidence...")
        final_verdicts = self._generate_verdicts(flows_with_ml)
        
        # Phase 7: Generate report
        report = self._generate_report(packets, flows, final_verdicts)
        
        logger.info("Analysis complete")
        
        return report
    
    def _extract_all_features(self, flows: List[Dict]) -> List[Dict]:
        """Extract all features from flows"""
        
        for flow in flows:
            try:
                # Statistical features
                if FEATURE_GROUPS.get("statistical"):
                    flow.update(statistical.StatisticalFeatures.extract_features(flow))
                
                # Timing features
                if FEATURE_GROUPS.get("timing"):
                    flow.update(timing.TimingFeatures.extract_features(flow))
                
                # DNS features
                if FEATURE_GROUPS.get("dns"):
                    flow.update(dns.DNSFeatures.extract_features(flow))
                
                # TLS features
                if FEATURE_GROUPS.get("tls"):
                    flow.update(tls.TLSFeatures.extract_features(flow))
                
                # QUIC features
                if FEATURE_GROUPS.get("quic"):
                    flow.update(quic.QUICFeatures.extract_features(flow))
            
            except Exception as e:
                logger.error(f"Error extracting features for flow {flow.get('flow_id')}: {e}")
        
        return flows
    
    def _run_ml_inference(self, flows: List[Dict]) -> List[Dict]:
        """Run ML models for threat detection"""
        
        if not self.is_trained:
            logger.warning("ML models not trained, skipping inference")
            return flows
        
        # Extract feature matrix
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Get feature names from first flow
        if not flows:
            return flows
        
        feature_keys = self._get_ml_feature_keys(flows[0])
        X = np.array([[flow.get(k, 0) for k in feature_keys] for flow in flows])
        
        # Run models
        for flow, idx in zip(flows, range(len(flows))):
            ml_scores = {}
            
            try:
                # Anomaly detection
                if self.anomaly_ensemble:
                    anomaly_scores = self.anomaly_ensemble.score(X[idx:idx+1])
                    ml_scores["anomaly_score"] = float(anomaly_scores[0])
                
                # C2 detection
                if self.c2_detector:
                    c2_scores = self.c2_detector.score(X[idx:idx+1])
                    ml_scores["c2_score"] = float(c2_scores[0])
                
                # Exfiltration detection
                if self.exfil_detector:
                    exfil_scores = self.exfil_detector.score(X[idx:idx+1])
                    ml_scores["exfil_score"] = float(exfil_scores[0])
                
                # Botnet detection
                if self.botnet_detector:
                    botnet_scores = self.botnet_detector.score(X[idx:idx+1])
                    ml_scores["botnet_score"] = float(botnet_scores[0])
                
                flow["ml_scores"] = ml_scores
            
            except Exception as e:
                logger.error(f"Error running ML inference: {e}")
                flow["ml_scores"] = {}
        
        return flows
    
    def _generate_verdicts(self, flows: List[Dict]) -> List[Dict]:
        """Generate verdicts using evidence correlation"""
        
        verdicts = []
        
        for flow in flows:
            try:
                verdict = self.evidence_correlator.correlate_evidence(
                    flow,
                    flow.get("ml_scores", {}),
                    flow  # Heuristic scores are in the flow dict
                )
                
                # Add flow info to verdict
                verdict["flow_id"] = str(flow["flow_id"])
                verdict["src_ip"] = flow["src_ip"]
                verdict["dst_ip"] = flow["dst_ip"]
                verdict["protocol"] = flow["protocol"]
                verdict["packet_count"] = flow.get("packet_count", 0)
                verdict["duration"] = flow.get("duration", 0)
                
                verdicts.append(verdict)
            
            except Exception as e:
                logger.error(f"Error generating verdict: {e}")
        
        return verdicts
    
    def _generate_report(self, packets: List[Dict], flows: List[Dict], verdicts: List[Dict]) -> Dict:
        """Generate analysis report"""
        
        # Filter high-risk flows
        high_risk = [v for v in verdicts if v.get("risk_score", 0) > 0.7]
        medium_risk = [v for v in verdicts if 0.5 < v.get("risk_score", 0) <= 0.7]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_packets": len(packets),
                "total_flows": len(flows),
                "high_risk_flows": len(high_risk),
                "medium_risk_flows": len(medium_risk),
                "overall_risk_level": "critical" if high_risk else "high" if medium_risk else "low"
            },
            "high_risk_findings": high_risk[:10],  # Top 10
            "medium_risk_findings": medium_risk[:10],
            "key_indicators": self._extract_key_indicators(flows, verdicts),
            "mitre_framework": self._aggregate_mitre_findings(verdicts),
            "recommendations": self._generate_recommendations(high_risk, medium_risk)
        }
        
        return report
    
    def _get_ml_feature_keys(self, flow: Dict) -> List[str]:
        """Get feature column names for ML models"""
        
        default_features = [
            'packet_size_mean', 'packet_size_std', 'packet_size_min', 'packet_size_max',
            'mean_iat', 'std_iat', 'burstiness',
            'dns_entropy', 'dns_unique_domains',
            'tls_entropy', 'tls_c2_score',
            'timing_periodicity', 'beaconing_score'
        ]
        
        # Return only features that exist in the flow
        return [f for f in default_features if f in flow]
    
    def _extract_key_indicators(self, flows: List[Dict], verdicts: List[Dict]) -> Dict:
        """Extract key security indicators"""
        
        indicators = {}
        
        # Protocol distribution
        protocols = {}
        for flow in flows:
            proto = flow.get("protocol", "unknown")
            protocols[proto] = protocols.get(proto, 0) + 1
        
        indicators["protocol_distribution"] = protocols
        
        # Top destinations
        destinations = {}
        for flow in flows:
            dst = flow.get("dst_ip")
            count = flow.get("packet_count", 0)
            if dst:
                destinations[dst] = destinations.get(dst, 0) + count
        
        top_dests = sorted(destinations.items(), key=lambda x: x[1], reverse=True)[:5]
        indicators["top_destinations"] = [{"ip": ip, "packets": pkt} for ip, pkt in top_dests]
        
        # Threat type distribution
        threat_types = {}
        for verdict in verdicts:
            for threat in verdict.get("threat_types", []):
                threat_types[threat] = threat_types.get(threat, 0) + 1
        
        indicators["threat_type_distribution"] = threat_types
        
        return indicators
    
    def _aggregate_mitre_findings(self, verdicts: List[Dict]) -> Dict:
        """Aggregate MITRE ATT&CK findings"""
        
        tactics = {}
        techniques = {}
        
        for verdict in verdicts:
            for tactic in verdict.get("mitre_tactics", []):
                tactics[tactic] = tactics.get(tactic, 0) + 1
        
        return {
            "top_tactics": sorted(tactics.items(), key=lambda x: x[1], reverse=True),
            "affected_flows": len([v for v in verdicts if v.get("mitre_tactics")])
        }
    
    def _generate_recommendations(self, high_risk: List[Dict], medium_risk: List[Dict]) -> List[str]:
        """Generate incident response recommendations"""
        
        recommendations = []
        
        if high_risk:
            recommendations.append("CRITICAL: Initiate incident response for high-risk flows")
            recommendations.append("Isolate affected internal systems immediately")
            
            # Type-specific recommendations
            for verdict in high_risk:
                if "C2_communication" in verdict.get("threat_types", []):
                    recommendations.append(f"Block outbound connections to {verdict['dst_ip']}")
                
                if "data_exfiltration" in verdict.get("threat_types", []):
                    recommendations.append("Monitor and preserve logs for forensic analysis")
        
        if medium_risk:
            recommendations.append("Enhanced monitoring recommended for medium-risk flows")
            recommendations.append("Consider threat hunt to identify root cause")
        
        recommendations.append("Review network segmentation policies")
        recommendations.append("Update threat intelligence feeds")
        
        return recommendations
    
    def train_models(self, training_data: Optional[str] = None):
        """Train ML models (requires labeled data)"""
        
        logger.info("Training ML models...")
        
        import numpy as np
        
        # Create ensemble
        self.anomaly_ensemble = AnomalyEnsemble(contamination=ANOMALY_CONTAMINATION)
        self.c2_detector = C2Detector()
        self.exfil_detector = DataExfilDetector()
        self.botnet_detector = BotnetDetector()
        
        # For now, create dummy training data
        # In production, this would load real labeled data
        n_samples = 100
        n_features = 20
        X_train = np.random.randn(n_samples, n_features)
        
        # Train models
        self.anomaly_ensemble.fit(X_train)
        self.c2_detector.fit(X_train)
        self.exfil_detector.fit(X_train)
        self.botnet_detector.fit(X_train)
        
        self.is_trained = True
        logger.info("Model training complete")


def main():
    """Example usage"""
    
    # Initialize pipeline
    pipeline = AnalysisPipeline()
    
    # Train models (optional)
    # pipeline.train_models()
    
    # Analyze PCAP
    # results = pipeline.analyze_pcap("data/raw_pcaps/sample.pcap", query="find C2 communications")
    
    # Print results
    # print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
