"""
Dashboard Package
FastAPI-based dashboard backend for AegisPCAP threat detection platform
"""

from .app import app
from .endpoints import router as dashboard_router
from .websocket import router as websocket_router, manager, broadcast_alert, broadcast_incident
from .schemas import (
    FlowSummary, FlowDetail, AlertSummary, AlertDetail,
    IncidentSummary, IncidentDetail, DashboardOverview,
    NetworkTopology, DashboardMetrics
)
from .config import get_config, config

__all__ = [
    "app",
    "dashboard_router",
    "websocket_router",
    "manager",
    "broadcast_alert",
    "broadcast_incident",
    "FlowSummary",
    "FlowDetail",
    "AlertSummary",
    "AlertDetail",
    "IncidentSummary",
    "IncidentDetail",
    "DashboardOverview",
    "NetworkTopology",
    "DashboardMetrics",
    "get_config",
    "config"
]

__version__ = "0.3.0"
__author__ = "AegisPCAP Team"
