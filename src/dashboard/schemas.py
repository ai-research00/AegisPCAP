"""
Dashboard Response Schemas
Pydantic models for dashboard API responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident investigation status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ============================================================================
# Pagination & Filtering
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=100, ge=1, le=1000, description="Items per page")
    sort_by: str = Field(default="timestamp", description="Field to sort by")
    sort_order: str = Field(default="desc", description="Sort order: asc or desc")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any] = Field(
        default_factory=lambda: {"page": 1, "page_size": 100, "total": 0}
    )
    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Flow Schemas
# ============================================================================

class GeoLocation(BaseModel):
    """Geographic location information"""
    country: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    asn: Optional[str] = None
    organization: Optional[str] = None


class FlowProtocolInfo(BaseModel):
    """Protocol-specific information"""
    protocol: str
    port: Optional[int] = None
    service: Optional[str] = None
    version: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class FlowSummary(BaseModel):
    """Summary of a network flow"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    packets: int
    bytes: int
    src_geo: Optional[GeoLocation] = None
    dst_geo: Optional[GeoLocation] = None
    risk_score: float = Field(default=0.0, ge=0, le=100)
    status: str = "active"


class FlowDetail(FlowSummary):
    """Detailed flow information"""
    packet_size_mean: Optional[float] = None
    packet_size_std: Optional[float] = None
    payload_entropy: Optional[float] = None
    retransmissions: Optional[int] = None
    dns_queries: Optional[List[str]] = None
    tls_fingerprint: Optional[str] = None
    user_agent: Optional[str] = None
    features: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


# ============================================================================
# Alert Schemas
# ============================================================================

class AlertBase(BaseModel):
    """Base alert information"""
    alert_id: str
    flow_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    description: str


class AlertSummary(AlertBase):
    """Summary of an alert"""
    detector: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    affected_ips: List[str] = Field(default_factory=list)


class AlertDetail(AlertSummary):
    """Detailed alert information"""
    mitre_techniques: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    related_alerts: List[str] = Field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# ============================================================================
# Incident Schemas
# ============================================================================

class IncidentBase(BaseModel):
    """Base incident information"""
    incident_id: str
    timestamp: datetime
    status: IncidentStatus
    title: str
    description: str


class IncidentSummary(IncidentBase):
    """Summary of an incident"""
    affected_flows: int
    severity: AlertSeverity
    alert_count: int
    risk_score: float = Field(default=0.0, ge=0, le=100)


class IncidentDetail(IncidentSummary):
    """Detailed incident information"""
    flows: List[str] = Field(default_factory=list, description="List of flow IDs")
    alerts: List[str] = Field(default_factory=list, description="List of alert IDs")
    affected_ips: List[str] = Field(default_factory=list)
    attack_vector: Optional[str] = None
    investigation_notes: str = ""
    assigned_to: Optional[str] = None
    started_at: datetime
    resolved_at: Optional[datetime] = None


# ============================================================================
# Analytics Schemas
# ============================================================================

class TimeSeriesPoint(BaseModel):
    """Single time series data point"""
    timestamp: datetime
    value: float
    label: Optional[str] = None


class TimeSeriesData(BaseModel):
    """Time series data for charts"""
    title: str
    unit: str
    data_points: List[TimeSeriesPoint]
    min_value: float
    max_value: float
    avg_value: float


class TopItem(BaseModel):
    """Top N item with count"""
    rank: int
    label: str
    value: float
    percentage: float = Field(default=0.0, ge=0, le=100)


class TopItemsList(BaseModel):
    """List of top N items"""
    title: str
    description: Optional[str] = None
    items: List[TopItem]
    total_count: int


class SystemStatistics(BaseModel):
    """System-wide statistics"""
    timestamp: datetime
    total_flows: int
    total_alerts: int
    total_incidents: int
    high_risk_flows: int
    active_incidents: int
    average_risk_score: float
    unique_source_ips: int
    unique_dest_ips: int


class DashboardMetrics(BaseModel):
    """Dashboard overview metrics"""
    stats: SystemStatistics
    threat_timeline: TimeSeriesData
    top_source_ips: TopItemsList
    top_dest_ips: TopItemsList
    top_protocols: TopItemsList
    alerts_by_severity: Dict[str, int]
    incidents_by_status: Dict[str, int]


# ============================================================================
# Dashboard Overview
# ============================================================================

class DashboardOverview(BaseModel):
    """Complete dashboard overview"""
    timestamp: datetime
    status: str = "healthy"
    metrics: DashboardMetrics
    recent_alerts: List[AlertSummary]
    recent_incidents: List[IncidentSummary]
    high_risk_flows: List[FlowSummary]


# ============================================================================
# Network Topology
# ============================================================================

class NetworkNode(BaseModel):
    """Node in network topology"""
    id: str
    label: str
    type: str  # "internal", "external", "server", "client"
    ip_address: str
    risk_score: float
    flow_count: int
    alert_count: int
    geo_location: Optional[GeoLocation] = None
    color: Optional[str] = None


class NetworkLink(BaseModel):
    """Link between nodes in network topology"""
    source_id: str
    target_id: str
    flow_count: int
    bytes_transferred: int
    alert_count: int
    risk_score: float
    color: Optional[str] = None


class NetworkTopology(BaseModel):
    """Network topology visualization data"""
    nodes: List[NetworkNode]
    links: List[NetworkLink]
    timestamp: datetime
    summary: Dict[str, Any]


# ============================================================================
# Threat Intelligence
# ============================================================================

class ThreatIndicator(BaseModel):
    """Single threat indicator"""
    type: str  # "ip", "domain", "hash", "url"
    value: str
    severity: AlertSeverity
    source: str
    first_seen: datetime
    last_seen: datetime
    matched_flows: int


class ThreatIntelligenceSummary(BaseModel):
    """Threat intelligence summary"""
    timestamp: datetime
    total_indicators: int
    indicators_by_type: Dict[str, int]
    recent_indicators: List[ThreatIndicator]


# ============================================================================
# Error Responses
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None


# ============================================================================
# Filter Parameters
# ============================================================================

class FlowFilterParams(BaseModel):
    """Flow filtering parameters"""
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    min_risk_score: float = Field(default=0, ge=0, le=100)
    max_risk_score: float = Field(default=100, ge=0, le=100)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    has_alerts: Optional[bool] = None


class AlertFilterParams(BaseModel):
    """Alert filtering parameters"""
    severity: Optional[AlertSeverity] = None
    detector: Optional[str] = None
    flow_id: Optional[str] = None
    min_confidence: float = Field(default=0, ge=0, le=1)
    acknowledged: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class IncidentFilterParams(BaseModel):
    """Incident filtering parameters"""
    status: Optional[IncidentStatus] = None
    severity: Optional[AlertSeverity] = None
    assigned_to: Optional[str] = None
    min_risk_score: float = Field(default=0, ge=0, le=100)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
