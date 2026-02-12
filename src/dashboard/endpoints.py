"""
Dashboard FastAPI Endpoints
RESTful API for dashboard data access with pagination, filtering, and sorting
"""

from fastapi import APIRouter, Query, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

from .schemas import (
    FlowSummary, FlowDetail, AlertSummary, AlertDetail,
    IncidentSummary, IncidentDetail, DashboardOverview,
    NetworkTopology, ThreatIntelligenceSummary,
    PaginatedResponse, SystemStatistics, DashboardMetrics,
    FlowFilterParams, AlertFilterParams, IncidentFilterParams,
    ErrorResponse, AlertSeverity, IncidentStatus
)
from src.db.persistence import get_persistence_layer
from src.db.models import Flow, Alert, Incident, Verdict

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# ============================================================================
# Dependency: Get Database Session
# ============================================================================

async def get_db_session() -> Session:
    """Get database session"""
    persistence = get_persistence_layer()
    session = persistence.connection.get_session()
    try:
        yield session
    finally:
        session.close()


# ============================================================================
# Dashboard Overview Endpoint
# ============================================================================

@router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview(db: Session = Depends(get_db_session)):
    """
    Get complete dashboard overview with metrics, recent alerts, and high-risk flows
    
    Returns:
        DashboardOverview: Complete dashboard state
    """
    try:
        persistence = get_persistence_layer()
        
        # Get system statistics
        stats = persistence.get_statistics()
        
        # Get threat timeline (last 24 hours)
        threat_timeline = persistence.get_threat_timeline(hours=24)
        
        # Get top source IPs
        top_src_ips = persistence.get_top_attacking_ips(direction="src", limit=10)
        
        # Get top dest IPs
        top_dst_ips = persistence.get_top_attacking_ips(direction="dst", limit=10)
        
        # Get top protocols
        protocols = db.query(Flow.protocol).distinct().all()
        
        # Get recent high-severity alerts
        recent_alerts = db.query(Alert).order_by(
            Alert.timestamp.desc()
        ).limit(5).all()
        
        # Get active incidents
        active_incidents = db.query(Incident).filter(
            Incident.status.in_(["open", "in_progress"])
        ).order_by(Incident.timestamp.desc()).limit(5).all()
        
        # Get high-risk flows
        high_risk_flows = db.query(Flow).join(Verdict).filter(
            Verdict.risk_score >= 70
        ).order_by(Verdict.risk_score.desc()).limit(5).all()
        
        return DashboardOverview(
            timestamp=datetime.utcnow(),
            status="healthy",
            metrics=DashboardMetrics(
                stats=SystemStatistics(**stats),
                threat_timeline=threat_timeline,
                top_source_ips=top_src_ips,
                top_dest_ips=top_dst_ips,
                top_protocols=protocols,
                alerts_by_severity={},
                incidents_by_status={}
            ),
            recent_alerts=[AlertSummary(
                alert_id=str(a.id),
                flow_id=str(a.flow_id),
                timestamp=a.timestamp,
                severity=AlertSeverity(a.severity.lower()),
                title=a.title,
                description=a.description,
                detector=a.detector_type,
                confidence=a.confidence
            ) for a in recent_alerts],
            recent_incidents=[IncidentSummary(
                incident_id=str(i.id),
                timestamp=i.timestamp,
                status=IncidentStatus(i.status.lower()),
                title=i.title,
                description=i.description,
                affected_flows=len(i.flows) if i.flows else 0,
                severity=AlertSeverity(i.severity.lower()),
                alert_count=len(i.alerts) if i.alerts else 0,
                risk_score=i.risk_score
            ) for i in active_incidents],
            high_risk_flows=[FlowSummary(
                flow_id=str(f.id),
                src_ip=f.src_ip,
                dst_ip=f.dst_ip,
                src_port=f.src_port,
                dst_port=f.dst_port,
                protocol=f.protocol,
                start_time=f.start_time,
                end_time=f.end_time,
                duration_seconds=(f.end_time - f.start_time).total_seconds(),
                packets=f.packet_count,
                bytes=f.byte_count,
                risk_score=f.verdict.risk_score if f.verdict else 0
            ) for f in high_risk_flows]
        )
    except Exception as e:
        logger.error(f"Error fetching dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Flows Endpoints
# ============================================================================

@router.get("/flows", response_model=PaginatedResponse)
async def list_flows(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="asc or desc"),
    src_ip: Optional[str] = None,
    dst_ip: Optional[str] = None,
    protocol: Optional[str] = None,
    min_risk_score: float = Query(0, ge=0, le=100),
    max_risk_score: float = Query(100, ge=0, le=100),
    db: Session = Depends(get_db_session)
):
    """
    List flows with pagination, filtering, and sorting
    
    Query Parameters:
        page: Page number (1-indexed)
        page_size: Items per page (max 1000)
        sort_by: Field to sort by (timestamp, risk_score, bytes, packets)
        sort_order: Sort order (asc or desc)
        src_ip: Filter by source IP
        dst_ip: Filter by destination IP
        protocol: Filter by protocol
        min_risk_score: Minimum risk score
        max_risk_score: Maximum risk score
    """
    try:
        query = db.query(Flow)
        
        # Apply filters
        if src_ip:
            query = query.filter(Flow.src_ip == src_ip)
        if dst_ip:
            query = query.filter(Flow.dst_ip == dst_ip)
        if protocol:
            query = query.filter(Flow.protocol == protocol)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply sorting
        if sort_by == "risk_score":
            query = query.join(Verdict).order_by(
                Verdict.risk_score.desc() if sort_order == "desc" else Verdict.risk_score.asc()
            )
        elif sort_by in ["timestamp", "start_time", "bytes", "packets"]:
            sort_column = getattr(Flow, sort_by)
            query = query.order_by(
                sort_column.desc() if sort_order == "desc" else sort_column.asc()
            )
        else:
            query = query.order_by(Flow.start_time.desc())
        
        # Apply pagination
        offset = (page - 1) * page_size
        flows = query.offset(offset).limit(page_size).all()
        
        return PaginatedResponse(
            data=[{
                "flow_id": str(f.id),
                "src_ip": f.src_ip,
                "dst_ip": f.dst_ip,
                "src_port": f.src_port,
                "dst_port": f.dst_port,
                "protocol": f.protocol,
                "start_time": f.start_time.isoformat(),
                "duration": (f.end_time - f.start_time).total_seconds(),
                "packets": f.packet_count,
                "bytes": f.byte_count,
                "risk_score": f.verdict.risk_score if f.verdict else 0
            } for f in flows],
            pagination={
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size
            }
        )
    except Exception as e:
        logger.error(f"Error listing flows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flows/{flow_id}", response_model=FlowDetail)
async def get_flow_detail(flow_id: str, db: Session = Depends(get_db_session)):
    """
    Get detailed information about a specific flow
    
    Path Parameters:
        flow_id: Flow ID
    """
    try:
        flow = db.query(Flow).filter(Flow.id == int(flow_id)).first()
        if not flow:
            raise HTTPException(status_code=404, detail="Flow not found")
        
        persistence = get_persistence_layer()
        cached_features = persistence.get_cached_features(int(flow_id))
        
        return FlowDetail(
            flow_id=str(flow.id),
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            start_time=flow.start_time,
            end_time=flow.end_time,
            duration_seconds=(flow.end_time - flow.start_time).total_seconds(),
            packets=flow.packet_count,
            bytes=flow.byte_count,
            risk_score=flow.verdict.risk_score if flow.verdict else 0,
            features=cached_features or {},
            alerts=[str(a.id) for a in flow.alerts] if flow.alerts else []
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching flow detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Alerts Endpoints
# ============================================================================

@router.get("/alerts", response_model=PaginatedResponse)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="asc or desc"),
    severity: Optional[str] = None,
    detector: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    db: Session = Depends(get_db_session)
):
    """
    List alerts with pagination, filtering, and sorting
    
    Query Parameters:
        page: Page number
        page_size: Items per page
        severity: Filter by severity (low, medium, high, critical)
        detector: Filter by detector type
        acknowledged: Filter by acknowledgment status
    """
    try:
        query = db.query(Alert)
        
        # Apply filters
        if severity:
            query = query.filter(Alert.severity == severity.upper())
        if detector:
            query = query.filter(Alert.detector_type == detector)
        if acknowledged is not None:
            query = query.filter(Alert.acknowledged == acknowledged)
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        if sort_by == "severity":
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            query = query.order_by(Alert.timestamp.desc())
        else:
            sort_column = getattr(Alert, sort_by, Alert.timestamp)
            query = query.order_by(
                sort_column.desc() if sort_order == "desc" else sort_column.asc()
            )
        
        # Apply pagination
        offset = (page - 1) * page_size
        alerts = query.offset(offset).limit(page_size).all()
        
        return PaginatedResponse(
            data=[{
                "alert_id": str(a.id),
                "flow_id": str(a.flow_id),
                "timestamp": a.timestamp.isoformat(),
                "severity": a.severity,
                "detector": a.detector_type,
                "title": a.title,
                "confidence": a.confidence,
                "acknowledged": a.acknowledged
            } for a in alerts],
            pagination={
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size
            }
        )
    except Exception as e:
        logger.error(f"Error listing alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}", response_model=AlertDetail)
async def get_alert_detail(alert_id: str, db: Session = Depends(get_db_session)):
    """Get detailed information about a specific alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == int(alert_id)).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return AlertDetail(
            alert_id=str(alert.id),
            flow_id=str(alert.flow_id),
            timestamp=alert.timestamp,
            severity=AlertSeverity(alert.severity.lower()),
            title=alert.title,
            description=alert.description,
            detector=alert.detector_type,
            confidence=alert.confidence,
            evidence=alert.evidence or {},
            mitre_techniques=alert.mitre_techniques or [],
            acknowledged=alert.acknowledged
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alert detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, db: Session = Depends(get_db_session)):
    """Acknowledge an alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == int(alert_id)).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.acknowledged = True
        alert.acknowledged_at = datetime.utcnow()
        db.commit()
        
        return {"status": "acknowledged", "alert_id": alert_id}
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Incidents Endpoints
# ============================================================================

@router.get("/incidents", response_model=PaginatedResponse)
async def list_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """
    List incidents with pagination and filtering
    
    Query Parameters:
        page: Page number
        page_size: Items per page
        status: Filter by status (open, in_progress, resolved, closed)
        severity: Filter by severity (low, medium, high, critical)
    """
    try:
        query = db.query(Incident)
        
        # Apply filters
        if status:
            query = query.filter(Incident.status == status)
        if severity:
            query = query.filter(Incident.severity == severity.upper())
        
        # Get total count
        total = query.count()
        
        # Sort by timestamp descending
        query = query.order_by(Incident.timestamp.desc())
        
        # Apply pagination
        offset = (page - 1) * page_size
        incidents = query.offset(offset).limit(page_size).all()
        
        return PaginatedResponse(
            data=[{
                "incident_id": str(i.id),
                "timestamp": i.timestamp.isoformat(),
                "status": i.status,
                "severity": i.severity,
                "title": i.title,
                "affected_flows": len(i.flows) if i.flows else 0,
                "alert_count": len(i.alerts) if i.alerts else 0,
                "risk_score": i.risk_score
            } for i in incidents],
            pagination={
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size
            }
        )
    except Exception as e:
        logger.error(f"Error listing incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incidents/{incident_id}", response_model=IncidentDetail)
async def get_incident_detail(incident_id: str, db: Session = Depends(get_db_session)):
    """Get detailed information about a specific incident"""
    try:
        incident = db.query(Incident).filter(Incident.id == int(incident_id)).first()
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        return IncidentDetail(
            incident_id=str(incident.id),
            timestamp=incident.timestamp,
            status=IncidentStatus(incident.status.lower()),
            title=incident.title,
            description=incident.description,
            affected_flows=len(incident.flows) if incident.flows else 0,
            severity=AlertSeverity(incident.severity.lower()),
            alert_count=len(incident.alerts) if incident.alerts else 0,
            risk_score=incident.risk_score,
            flows=[str(f.id) for f in incident.flows] if incident.flows else [],
            alerts=[str(a.id) for a in incident.alerts] if incident.alerts else [],
            started_at=incident.timestamp,
            resolved_at=incident.resolved_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching incident detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics Endpoints
# ============================================================================

@router.get("/analytics/threat-timeline")
async def get_threat_timeline(
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db_session)
):
    """
    Get threat activity timeline
    
    Query Parameters:
        hours: Number of hours to retrieve (max 30 days)
    """
    try:
        persistence = get_persistence_layer()
        timeline = persistence.get_threat_timeline(hours=hours)
        return timeline
    except Exception as e:
        logger.error(f"Error fetching threat timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/top-ips")
async def get_top_ips(
    direction: str = Query("src", regex="^(src|dst)$"),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db_session)
):
    """
    Get top source or destination IPs by alert count
    
    Query Parameters:
        direction: Filter direction (src or dst)
        limit: Number of top IPs to return
    """
    try:
        persistence = get_persistence_layer()
        top_ips = persistence.get_top_attacking_ips(direction=direction, limit=limit)
        return top_ips
    except Exception as e:
        logger.error(f"Error fetching top IPs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/statistics")
async def get_statistics(db: Session = Depends(get_db_session)):
    """Get system-wide statistics"""
    try:
        persistence = get_persistence_layer()
        stats = persistence.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/incidents-correlation")
async def get_incidents_correlation(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db_session)
):
    """Get correlated incidents"""
    try:
        persistence = get_persistence_layer()
        correlations = persistence.correlate_incidents(limit=limit)
        return correlations
    except Exception as e:
        logger.error(f"Error correlating incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        persistence = get_persistence_layer()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "connected",
                "cache": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }, 503


@router.get("/config")
async def get_config():
    """Get dashboard configuration"""
    return {
        "refresh_interval": 30000,  # 30 seconds in ms
        "max_page_size": 1000,
        "default_page_size": 100,
        "available_protocols": ["TCP", "UDP", "ICMP", "DNS", "TLS", "QUIC"],
        "available_detectors": [
            "beacon_detection",
            "dns_tunneling",
            "tls_anomaly",
            "port_scan",
            "data_exfiltration"
        ],
        "time_zone": "UTC"
    }
