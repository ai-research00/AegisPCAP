"""
Advanced Task Planner - Intent recognition and tool orchestration for multi-turn analysis
"""
import logging
from typing import List, Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Plan and decompose complex security queries"""
    
    # Intent types
    INTENTS = {
        'hunt': 'Investigate flows for potential threats',
        'investigate': 'Deep-dive analysis of specific flow/IP',
        'detect_c2': 'Detect command and control communications',
        'detect_exfil': 'Find data exfiltration patterns',
        'detect_lateral': 'Detect lateral movement',
        'detect_port_scan': 'Identify port scanning activity',
        'detect_ddos': 'Detect DDoS attack patterns',
        'detect_beaconing': 'Find beaconing/periodic communication',
        'threat_intelligence': 'Correlate with threat intel',
        'incident_timeline': 'Build forensic timeline',
        'risk_assessment': 'Overall risk assessment',
        'report': 'Generate security report'
    }
    
    # Tools available
    TOOLS = {
        'anomaly_detector': 'Generic anomaly detection',
        'c2_detector': 'Specialized C2 detection',
        'exfil_detector': 'Data exfiltration detection',
        'botnet_detector': 'Botnet behavior detection',
        'dga_detector': 'Domain Generation Algorithm detection',
        'dns_analyzer': 'DNS heuristics analysis',
        'tls_analyzer': 'TLS fingerprint analysis',
        'timing_analyzer': 'Timing pattern analysis',
        'flow_aggregator': 'Flow aggregation and correlation',
        'threat_intel_lookup': 'Threat intelligence correlation',
        'whois_lookup': 'WHOIS and ASN information',
        'geo_analyzer': 'Geographic analysis',
        'evidence_correlator': 'Multi-source evidence correlation'
    }
    
    def __init__(self):
        self.query_history = []
        self.context = {}
    
    def plan(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Plan task decomposition from natural language query
        
        Args:
            query: User query or task
            context: Contextual information (previous findings, etc.)
            
        Returns:
            Task plan with tools and reasoning
        """
        
        if context:
            self.context = context
        
        # Extract intent
        intent = self._recognize_intent(query)
        
        # Extract entities (IPs, domains, ports, time ranges)
        entities = self._extract_entities(query)
        
        # Select tools
        tools = self._select_tools(intent, entities)
        
        # Create task plan
        plan = {
            "query": query,
            "intent": intent,
            "entities": entities,
            "tools": tools,
            "reasoning": self._generate_reasoning(intent, entities, tools),
            "execution_order": self._optimize_execution_order(tools),
            "expected_outputs": self._predict_outputs(intent, tools),
            "follow_up_questions": self._suggest_follow_ups(intent, entities)
        }
        
        return plan
    
    def _recognize_intent(self, query: str) -> str:
        """Recognize query intent"""
        
        query_lower = query.lower()
        
        # Intent keyword matching
        intent_keywords = {
            'hunt': ['hunt', 'find', 'search', 'look for', 'identify', 'discover'],
            'detect_c2': ['c2', 'command and control', 'command&control', 'beacon', 'periodic'],
            'detect_exfil': ['exfiltration', 'exfil', 'data loss', 'data out', 'outbound'],
            'detect_lateral': ['lateral', 'movement', 'propagat', 'spread', 'internal'],
            'detect_port_scan': ['port scan', 'scanning', 'enumerat', 'discover port'],
            'detect_beaconing': ['beacon', 'periodic', 'regular interval', 'timing'],
            'threat_intelligence': ['threat intel', 'reputation', 'known bad'],
            'incident_timeline': ['timeline', 'forensic', 'sequence', 'chain'],
            'risk_assessment': ['risk', 'severity', 'impact', 'critical'],
            'report': ['report', 'summary', 'findings', 'conclusion']
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_intents.append(intent)
        
        # Default to hunt if no specific intent
        return detected_intents[0] if detected_intents else 'hunt'
    
    def _extract_entities(self, query: str) -> Dict:
        """Extract entities (IPs, domains, ports, time ranges)"""
        
        entities = {
            "ips": [],
            "domains": [],
            "ports": [],
            "time_ranges": [],
            "protocols": []
        }
        
        # IP addresses (simple regex)
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        entities["ips"] = re.findall(ip_pattern, query)
        
        # Domains
        domain_pattern = r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b'
        entities["domains"] = re.findall(domain_pattern, query, re.IGNORECASE)
        
        # Ports
        port_pattern = r'(?:port|:)\s*(\d{1,5})'
        entities["ports"] = [int(p) for p in re.findall(port_pattern, query)]
        
        # Time ranges
        time_keywords = {
            'last_hour': 3600,
            'last_day': 86400,
            'last_week': 604800,
            'last_month': 2592000
        }
        
        for keyword, seconds in time_keywords.items():
            if keyword in query.lower():
                entities["time_ranges"].append(keyword)
        
        # Protocols
        for protocol in ['TCP', 'UDP', 'DNS', 'TLS', 'QUIC', 'HTTP', 'HTTPS']:
            if protocol in query.upper():
                entities["protocols"].append(protocol)
        
        return entities
    
    def _select_tools(self, intent: str, entities: Dict) -> List[str]:
        """Select appropriate tools for intent"""
        
        tools = []
        
        # Base on intent
        if intent == 'detect_c2':
            tools = ['c2_detector', 'tls_analyzer', 'dns_analyzer', 'timing_analyzer']
        
        elif intent == 'detect_exfil':
            tools = ['exfil_detector', 'flow_aggregator', 'geo_analyzer']
        
        elif intent == 'detect_beaconing':
            tools = ['botnet_detector', 'timing_analyzer', 'dns_analyzer']
        
        elif intent == 'detect_port_scan':
            tools = ['flow_aggregator', 'anomaly_detector']
        
        elif intent == 'hunt':
            tools = ['anomaly_detector', 'dga_detector', 'evidence_correlator']
        
        elif intent == 'threat_intelligence':
            tools = ['threat_intel_lookup', 'whois_lookup', 'geo_analyzer']
        
        elif intent == 'incident_timeline':
            tools = ['flow_aggregator', 'evidence_correlator']
        
        elif intent == 'report':
            tools = ['evidence_correlator', 'threat_intel_lookup']
        
        else:
            tools = ['anomaly_detector', 'evidence_correlator']
        
        # Add specialized tools based on entities
        if entities["domains"]:
            tools.insert(0, 'dns_analyzer')
        
        if entities["time_ranges"]:
            tools.append('timing_analyzer')
        
        # Always add evidence correlator
        if 'evidence_correlator' not in tools:
            tools.append('evidence_correlator')
        
        return tools
    
    def _optimize_execution_order(self, tools: List[str]) -> List[str]:
        """Optimize tool execution order (dependency aware)"""
        
        # Define dependencies
        dependencies = {
            'evidence_correlator': ['anomaly_detector', 'c2_detector', 'exfil_detector'],
            'threat_intel_lookup': ['flow_aggregator'],
            'whois_lookup': ['flow_aggregator']
        }
        
        ordered = []
        remaining = set(tools)
        
        while remaining:
            # Find tools with no remaining dependencies
            ready = []
            for tool in remaining:
                deps = dependencies.get(tool, [])
                if all(dep in ordered for dep in deps):
                    ready.append(tool)
            
            if not ready:
                # Add any remaining to break cycle
                ready = [remaining.pop()]
            
            ordered.extend(sorted(ready))
            remaining -= set(ready)
        
        return ordered
    
    def _generate_reasoning(self, intent: str, entities: Dict, tools: List[str]) -> str:
        """Generate explanation of plan"""
        
        reasoning = f"Intent: {intent}. "
        
        if entities["ips"]:
            reasoning += f"Analyzing IPs: {', '.join(entities['ips'])}. "
        
        if entities["domains"]:
            reasoning += f"Analyzing domains: {', '.join(entities['domains'])}. "
        
        reasoning += f"Using tools: {', '.join(tools[:3])}. "
        
        return reasoning
    
    def _predict_outputs(self, intent: str, tools: List[str]) -> List[str]:
        """Predict expected outputs"""
        
        outputs = [
            "Anomaly scores",
            "Risk assessment",
            "Threat classification"
        ]
        
        if 'dns_analyzer' in tools:
            outputs.append("DNS indicators (entropy, beaconing)")
        
        if 'tls_analyzer' in tools:
            outputs.append("TLS fingerprints and certificates")
        
        if 'threat_intel_lookup' in tools:
            outputs.append("Threat intelligence matches")
        
        return outputs
    
    def _suggest_follow_ups(self, intent: str, entities: Dict) -> List[str]:
        """Suggest follow-up questions"""
        
        follow_ups = []
        
        if intent == 'detect_c2':
            follow_ups.append("What is the suspected command infrastructure?")
            follow_ups.append("How long has this communication been active?")
        
        elif intent == 'detect_exfil':
            follow_ups.append("What data is at risk?")
            follow_ups.append("Where is the data being sent?")
        
        if entities["ips"]:
            follow_ups.append(f"Should we block {entities['ips'][0]}?")
        
        if entities["domains"]:
            follow_ups.append(f"Is {entities['domains'][0]} on our whitelist?")
        
        return follow_ups


def plan(query: str) -> List[str]:
    """Legacy function for backward compatibility"""
    planner = TaskPlanner()
    task_plan = planner.plan(query)
    return task_plan.get('tools', ['anomaly'])
