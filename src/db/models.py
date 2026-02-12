"""
AegisPCAP Database Models - SQLAlchemy ORM for persistent storage
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import Optional, List
import json

Base = declarative_base()


class Flow(Base):
    """Network flow record"""
    __tablename__ = 'flows'
    
    id = Column(Integer, primary_key=True)
    
    # Flow identification
    flow_hash = Column(String(64), unique=True, nullable=False, index=True)  # MD5 of flow_id
    src_ip = Column(String(45), nullable=False, index=True)  # IPv4/IPv6
    dst_ip = Column(String(45), nullable=False, index=True)
    src_port = Column(Integer)
    dst_port = Column(Integer)
    protocol = Column(String(10), nullable=False)  # TCP/UDP/ICMP/etc
    
    # Temporal
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    duration = Column(Float)  # seconds
    
    # Volume metrics
    packet_count = Column(Integer)
    total_bytes = Column(Integer)
    fwd_packets = Column(Integer)
    bwd_packets = Column(Integer)
    fwd_bytes = Column(Integer)
    bwd_bytes = Column(Integer)
    
    # Geolocation
    src_country = Column(String(2))
    src_city = Column(String(100))
    dst_country = Column(String(2))
    dst_city = Column(String(100))
    
    # Application layer
    dns_queries = Column(JSON)  # List of DNS queries
    tls_snis = Column(JSON)  # List of TLS SNIs
    http_hosts = Column(JSON)
    
    # Quality metrics
    malformed_packets = Column(Integer, default=0)
    retransmissions = Column(Integer, default=0)
    
    # Analysis metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String(10))  # Version of analysis pipeline
    
    # Relationships
    features = relationship('FlowFeatures', back_populates='flow', cascade='all, delete-orphan')
    alerts = relationship('Alert', back_populates='flow', cascade='all, delete-orphan')
    verdicts = relationship('Verdict', back_populates='flow', cascade='all, delete-orphan')
    
    # Indices
    __table_args__ = (
        Index('idx_flow_ips', 'src_ip', 'dst_ip'),
        Index('idx_flow_time', 'start_time', 'end_time'),
        Index('idx_flow_protocol', 'protocol'),
    )
    
    def __repr__(self):
        return f"<Flow {self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port} {self.protocol}>"


class FlowFeatures(Base):
    """Extracted features for machine learning"""
    __tablename__ = 'flow_features'
    
    id = Column(Integer, primary_key=True)
    flow_id = Column(Integer, ForeignKey('flows.id'), nullable=False, index=True)
    
    # Statistical features
    pkt_size_mean = Column(Float)
    pkt_size_std = Column(Float)
    pkt_size_min = Column(Float)
    pkt_size_max = Column(Float)
    pkt_size_median = Column(Float)
    pkt_size_skewness = Column(Float)
    pkt_size_kurtosis = Column(Float)
    pkt_size_cv = Column(Float)
    upload_download_ratio = Column(Float)
    
    # Timing features
    mean_iat = Column(Float)  # Inter-arrival time
    std_iat = Column(Float)
    min_iat = Column(Float)
    max_iat = Column(Float)
    median_iat = Column(Float)
    burstiness = Column(Float)
    timing_periodicity = Column(Float)
    packet_arrival_regularity = Column(Float)
    beaconing_score = Column(Float)
    
    # DNS features
    dns_entropy = Column(Float)
    dns_entropy_max = Column(Float)
    dns_entropy_min = Column(Float)
    dns_unique_domains = Column(Integer)
    dns_query_uniqueness = Column(Float)
    dns_dga_score = Column(Float)
    dns_beaconing_score = Column(Float)
    dns_tunnel_score = Column(Float)
    
    # TLS features
    tls_entropy = Column(Float)
    tls_entropy_max = Column(Float)
    tls_unique_snis = Column(Integer)
    tls_sni_uniqueness = Column(Float)
    tls_self_signed_score = Column(Float)
    tls_reuse_score = Column(Float)
    tls_c2_score = Column(Float)
    tls_certificate_diversity = Column(Float)
    
    # QUIC features
    quic_score = Column(Float)
    quic_packet_sizes_variance = Column(Float)
    quic_size_consistency = Column(Float)
    
    # TCP features
    syn_count = Column(Integer)
    fin_count = Column(Integer)
    rst_count = Column(Integer)
    ack_count = Column(Integer)
    
    # Entropy
    payload_entropy_mean = Column(Float)
    payload_entropy_std = Column(Float)
    
    # Computed at
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    flow = relationship('Flow', back_populates='features')
    
    __table_args__ = (
        Index('idx_flow_features_flow', 'flow_id'),
    )


class Alert(Base):
    """Security alerts/detections"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    flow_id = Column(Integer, ForeignKey('flows.id'), nullable=False, index=True)
    
    # Alert metadata
    alert_type = Column(String(50), nullable=False, index=True)  # C2, exfil, botnet, DGA, etc
    severity = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    risk_score = Column(Float, nullable=False)  # 0-1
    confidence = Column(Float)  # 0-1
    
    # Detection details
    detection_source = Column(String(100))  # ML model or heuristic name
    evidence = Column(JSON)  # List of evidence items
    mitre_tactics = Column(JSON)  # MITRE ATT&CK tactics
    mitre_techniques = Column(JSON)  # MITRE ATT&CK techniques
    
    # Status
    status = Column(String(20), default='new')  # new, investigating, confirmed, false_positive, resolved
    notes = Column(Text)
    
    # Temporal
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Investigation
    investigator = Column(String(100))  # Who investigated
    closed_at = Column(DateTime)
    
    # Relationship
    flow = relationship('Flow', back_populates='alerts')
    
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_alert_detected', 'detected_at'),
        Index('idx_alert_status', 'status'),
    )


class Verdict(Base):
    """Analysis verdicts/results"""
    __tablename__ = 'verdicts'
    
    id = Column(Integer, primary_key=True)
    flow_id = Column(Integer, ForeignKey('flows.id'), nullable=False, index=True)
    
    # Overall assessment
    risk_score = Column(Float, nullable=False)  # 0-100
    risk_level = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW, BENIGN
    confidence = Column(Float)  # 0-1
    
    # Threat classification
    threat_types = Column(JSON)  # List of detected threat types
    mitre_tactics = Column(JSON)  # List of tactics
    
    # False positive likelihood
    fp_likelihood = Column(Float)  # 0-1
    
    # Recommendation
    recommended_action = Column(String(50))  # block, isolate, monitor, investigate, allow
    notes = Column(Text)
    
    # Analysis details
    ml_scores = Column(JSON)  # Individual model scores
    heuristic_scores = Column(JSON)  # Heuristic scores
    
    # Temporal
    analyzed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Relationship
    flow = relationship('Flow', back_populates='verdicts')
    
    __table_args__ = (
        Index('idx_verdict_risk', 'risk_score'),
        Index('idx_verdict_risk_level', 'risk_level'),
    )


class ThreatIntelligence(Base):
    """Threat intelligence data (IPs, domains, file hashes)"""
    __tablename__ = 'threat_intelligence'
    
    id = Column(Integer, primary_key=True)
    
    # Entity information
    entity_type = Column(String(20), nullable=False)  # ip, domain, hash, cert, etc
    entity_value = Column(String(500), nullable=False, index=True)
    
    # Threat data
    threat_source = Column(String(100))  # VirusTotal, AlienVault OTX, etc
    threat_score = Column(Float)  # Reputation score (0-1)
    threat_category = Column(String(100))  # malware, botnet, c2, etc
    
    # Detection counts
    detections = Column(Integer)  # Number of engines detecting as malicious
    total_engines = Column(Integer)  # Total engines scanned
    
    # Metadata
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    source_url = Column(String(500))
    
    # Temporal
    imported_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_ti_entity', 'entity_type', 'entity_value'),
        Index('idx_ti_threat_score', 'threat_score'),
    )


class Incident(Base):
    """Security incidents (grouping related alerts/flows)"""
    __tablename__ = 'incidents'
    
    id = Column(Integer, primary_key=True)
    
    # Incident metadata
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Classification
    incident_type = Column(String(50), nullable=False)  # breach, c2, malware, etc
    severity = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    status = Column(String(20), default='open')  # open, investigating, contained, resolved, false_positive
    
    # Scope
    affected_ips = Column(JSON)  # List of affected IPs
    affected_users = Column(JSON)  # List of affected users
    
    # Temporal
    first_detected = Column(DateTime, nullable=False, index=True)
    last_activity = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Investigation
    investigator = Column(String(100))
    notes = Column(Text)
    
    # Related data
    related_alerts = Column(JSON)  # Alert IDs
    related_flows = Column(JSON)  # Flow IDs
    
    __table_args__ = (
        Index('idx_incident_type', 'incident_type'),
        Index('idx_incident_severity', 'severity'),
        Index('idx_incident_status', 'status'),
        Index('idx_incident_detected', 'first_detected'),
    )


class AuditLog(Base):
    """Audit trail for all analysis and actions"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    
    # Action info
    action = Column(String(100), nullable=False)  # analyze, train, block, etc
    entity_type = Column(String(50))  # flow, incident, alert, etc
    entity_id = Column(Integer)
    
    # User/system
    actor = Column(String(100))  # username or system
    source_ip = Column(String(45))
    
    # Details
    details = Column(JSON)
    result = Column(String(20))  # success, failure
    error_message = Column(Text)
    
    # Temporal
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_audit_action', 'action'),
        Index('idx_audit_actor', 'actor'),
        Index('idx_audit_timestamp', 'timestamp'),
    )


class ModelMetadata(Base):
    """ML Model registry and version tracking"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    
    # Model info
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # ensemble, c2_detector, etc
    version = Column(String(20), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    
    # Training info
    trained_on = Column(Integer)  # Number of samples
    training_date = Column(DateTime)
    training_duration = Column(Float)  # seconds
    
    # Deployment
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime)
    
    # Metadata
    hyperparameters = Column(JSON)
    feature_importance = Column(JSON)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'version'),
        Index('idx_model_active', 'is_active'),
    )


class FeatureStore(Base):
    """Pre-computed feature cache for fast queries"""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True)
    
    # Time window
    time_window = Column(String(50), nullable=False)  # e.g., "2026-02-04_10:00"
    
    # Features (pre-aggregated)
    entity_ip = Column(String(45), nullable=False, index=True)
    entity_domain = Column(String(255), index=True)
    
    # Aggregated metrics
    flow_count = Column(Integer)
    unique_destinations = Column(Integer)
    unique_domains = Column(Integer)
    total_bytes = Column(Integer)
    avg_risk_score = Column(Float)
    
    # Feature vector (JSON for flexibility)
    features = Column(JSON)
    
    # Temporal
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)
    ttl_seconds = Column(Integer, default=3600)  # Time-to-live
    
    __table_args__ = (
        Index('idx_feature_store_time', 'time_window'),
        Index('idx_feature_store_entity', 'entity_ip', 'entity_domain'),
    )


# Database session management
class DatabaseSession:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False, pool_size=20, max_overflow=40)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def init_db(self):
        """Initialize database schema"""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Repository pattern for data access
class FlowRepository:
    """Data access layer for flows"""
    
    def __init__(self, session):
        self.session = session
    
    def create(self, flow_data: dict) -> Flow:
        """Create new flow record"""
        flow = Flow(**flow_data)
        self.session.add(flow)
        self.session.commit()
        return flow
    
    def get_by_hash(self, flow_hash: str) -> Optional[Flow]:
        """Get flow by hash"""
        return self.session.query(Flow).filter(Flow.flow_hash == flow_hash).first()
    
    def get_by_id(self, flow_id: int) -> Optional[Flow]:
        """Get flow by ID"""
        return self.session.query(Flow).filter(Flow.id == flow_id).first()
    
    def get_risky_flows(self, min_risk_score: float = 0.7, limit: int = 100) -> List[Flow]:
        """Get high-risk flows"""
        return self.session.query(Flow).join(Verdict).filter(
            Verdict.risk_score >= min_risk_score
        ).limit(limit).all()
    
    def get_by_ip_pair(self, src_ip: str, dst_ip: str, limit: int = 10) -> List[Flow]:
        """Get flows between two IPs"""
        return self.session.query(Flow).filter(
            Flow.src_ip == src_ip,
            Flow.dst_ip == dst_ip
        ).limit(limit).all()
    
    def get_recent(self, hours: int = 24, limit: int = 1000) -> List[Flow]:
        """Get recent flows"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.session.query(Flow).filter(Flow.start_time >= cutoff).limit(limit).all()


class AlertRepository:
    """Data access layer for alerts"""
    
    def __init__(self, session):
        self.session = session
    
    def create(self, alert_data: dict) -> Alert:
        """Create new alert"""
        alert = Alert(**alert_data)
        self.session.add(alert)
        self.session.commit()
        return alert
    
    def get_critical_alerts(self, limit: int = 50) -> List[Alert]:
        """Get critical severity alerts"""
        return self.session.query(Alert).filter(
            Alert.severity == 'CRITICAL'
        ).order_by(Alert.detected_at.desc()).limit(limit).all()
    
    def get_by_status(self, status: str, limit: int = 100) -> List[Alert]:
        """Get alerts by status"""
        return self.session.query(Alert).filter(Alert.status == status).limit(limit).all()


class IncidentRepository:
    """Data access layer for incidents"""
    
    def __init__(self, session):
        self.session = session
    
    def create(self, incident_data: dict) -> Incident:
        """Create new incident"""
        incident = Incident(**incident_data)
        self.session.add(incident)
        self.session.commit()
        return incident
    
    def get_open_incidents(self, limit: int = 100) -> List[Incident]:
        """Get open incidents"""
        return self.session.query(Incident).filter(Incident.status == 'open').limit(limit).all()
    
    def get_critical_incidents(self, limit: int = 50) -> List[Incident]:
        """Get critical incidents"""
        return self.session.query(Incident).filter(
            Incident.severity == 'CRITICAL'
        ).order_by(Incident.first_detected.desc()).limit(limit).all()
