"""
Community Research API

Extends Phase 14 Research API with community features:
- Query anonymized data with access control
- Request data access for restricted datasets
- Dataset discovery and metadata
- Rate limiting and quota management
- Audit logging for compliance

Type hints: 100% coverage
Docstrings: 100% coverage
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib

from src.research.research_api import ResearchAPIController, QueryResponse


# ============================================================================
# ENUMS
# ============================================================================

class AccessTier(Enum):
    """User access tiers with different quotas."""
    PUBLIC = "public"  # 100 queries/day
    ACADEMIC = "academic"  # 1000 queries/day
    ENTERPRISE = "enterprise"  # 10000 queries/day
    UNLIMITED = "unlimited"  # No limits


class DatasetAccessLevel(Enum):
    """Dataset access levels."""
    PUBLIC = "public"  # Anyone can access
    RESTRICTED = "restricted"  # Requires approval
    PRIVATE = "private"  # Internal only


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UserQuota:
    """User quota tracking."""
    user_id: str
    tier: AccessTier
    queries_today: int = 0
    last_reset: str = field(default_factory=lambda: datetime.utcnow().date().isoformat())
    total_queries: int = 0
    
    def get_daily_limit(self) -> int:
        """Get daily query limit for tier."""
        limits = {
            AccessTier.PUBLIC: 100,
            AccessTier.ACADEMIC: 1000,
            AccessTier.ENTERPRISE: 10000,
            AccessTier.UNLIMITED: float('inf')
        }
        return limits.get(self.tier, 100)
    
    def is_within_quota(self) -> bool:
        """Check if user is within quota."""
        # Reset counter if new day
        today = datetime.utcnow().date().isoformat()
        if self.last_reset != today:
            self.queries_today = 0
            self.last_reset = today
        
        return self.queries_today < self.get_daily_limit()
    
    def increment_usage(self) -> None:
        """Increment query counter."""
        self.queries_today += 1
        self.total_queries += 1


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    dataset_id: str
    name: str
    description: str
    access_level: DatasetAccessLevel
    record_count: int
    feature_count: int
    labels_available: bool
    created_date: str
    last_updated: str
    citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "access_level": self.access_level.value,
            "record_count": self.record_count,
            "feature_count": self.feature_count,
            "labels_available": self.labels_available,
            "created_date": self.created_date,
            "last_updated": self.last_updated,
            "citation": self.citation
        }


@dataclass
class AccessRequest:
    """Data access request."""
    request_id: str
    user_id: str
    dataset_id: str
    purpose: str
    institution: str
    status: str = "pending"  # pending, approved, denied
    requested_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reviewed_date: Optional[str] = None
    reviewer_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "purpose": self.purpose,
            "institution": self.institution,
            "status": self.status,
            "requested_date": self.requested_date,
            "reviewed_date": self.reviewed_date,
            "reviewer_notes": self.reviewer_notes
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for data access."""
    log_id: str
    user_id: str
    query_type: str
    dataset_id: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    records_returned: int = 0
    anonymized: bool = True
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_id": self.log_id,
            "user_id": self.user_id,
            "query_type": self.query_type,
            "dataset_id": self.dataset_id,
            "timestamp": self.timestamp,
            "records_returned": self.records_returned,
            "anonymized": self.anonymized,
            "ip_address": self.ip_address
        }


# ============================================================================
# COMMUNITY RESEARCH API
# ============================================================================

class CommunityResearchAPI:
    """
    Community Research API extending Phase 14 Research API.
    
    Provides:
    - Anonymized data access with PII removal
    - Rate limiting and quota management
    - Access control for restricted datasets
    - Audit logging for compliance
    """
    
    def __init__(self):
        """Initialize community research API."""
        self.logger = logging.getLogger(__name__)
        self.base_api = ResearchAPIController()
        
        # User quota tracking
        self.user_quotas: Dict[str, UserQuota] = {}
        
        # Dataset registry
        self.datasets: Dict[str, DatasetInfo] = {}
        self._initialize_datasets()
        
        # Access requests
        self.access_requests: Dict[str, AccessRequest] = {}
        
        # Audit log
        self.audit_log: List[AuditLogEntry] = []
    
    def _initialize_datasets(self) -> None:
        """Initialize available datasets."""
        # Public benchmark datasets
        self.datasets["cicids2017"] = DatasetInfo(
            dataset_id="cicids2017",
            name="CICIDS2017",
            description="Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017",
            access_level=DatasetAccessLevel.PUBLIC,
            record_count=2830000,
            feature_count=78,
            labels_available=True,
            created_date="2017-07-01",
            last_updated="2017-07-01",
            citation="Sharafaldin et al., 2018"
        )
        
        self.datasets["unsw_nb15"] = DatasetInfo(
            dataset_id="unsw_nb15",
            name="UNSW-NB15",
            description="University of New South Wales Network-Based 2015 Dataset",
            access_level=DatasetAccessLevel.PUBLIC,
            record_count=2540047,
            feature_count=42,
            labels_available=True,
            created_date="2015-01-01",
            last_updated="2015-12-31",
            citation="Moustafa & Slay, 2015"
        )
        
        # Restricted internal dataset
        self.datasets["internal_production"] = DatasetInfo(
            dataset_id="internal_production",
            name="Internal Production Network",
            description="Real-world production network traffic (anonymized)",
            access_level=DatasetAccessLevel.RESTRICTED,
            record_count=15000000,
            feature_count=56,
            labels_available=True,
            created_date="2025-01-01",
            last_updated="2026-02-12",
            citation=None
        )
    
    def _get_or_create_quota(self, user_id: str, tier: AccessTier = AccessTier.PUBLIC) -> UserQuota:
        """Get or create user quota."""
        if user_id not in self.user_quotas:
            self.user_quotas[user_id] = UserQuota(user_id=user_id, tier=tier)
        return self.user_quotas[user_id]
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        quota = self._get_or_create_quota(user_id)
        return quota.is_within_quota()
    
    def _log_access(
        self,
        user_id: str,
        query_type: str,
        dataset_id: Optional[str],
        records_returned: int,
        anonymized: bool,
        ip_address: Optional[str] = None
    ) -> None:
        """Log data access for audit."""
        log_id = hashlib.sha256(
            f"{user_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = AuditLogEntry(
            log_id=log_id,
            user_id=user_id,
            query_type=query_type,
            dataset_id=dataset_id,
            records_returned=records_returned,
            anonymized=anonymized,
            ip_address=ip_address
        )
        
        self.audit_log.append(entry)
        self.logger.info(f"Audit log: {user_id} accessed {query_type} ({records_returned} records)")
    
    def query_anonymized_data(
        self,
        user_id: str,
        query_type: str,
        dataset_id: str = "internal",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query anonymized data with access control.
        
        Args:
            user_id: User identifier
            query_type: Type of query (flows, threats, statistics)
            dataset_id: Dataset to query
            filters: Optional query filters
            limit: Maximum records to return
            ip_address: Client IP address for audit
            
        Returns:
            Query results with anonymized data
        """
        # Check rate limit
        if not self._check_rate_limit(user_id):
            quota = self._get_or_create_quota(user_id)
            return {
                "error": "rate_limit_exceeded",
                "message": f"Daily quota exceeded ({quota.get_daily_limit()} queries/day)",
                "quota_reset": (datetime.utcnow() + timedelta(days=1)).date().isoformat()
            }
        
        # Check dataset access
        if dataset_id not in self.datasets:
            return {
                "error": "dataset_not_found",
                "message": f"Dataset '{dataset_id}' does not exist"
            }
        
        dataset = self.datasets[dataset_id]
        
        # Check access level
        if dataset.access_level == DatasetAccessLevel.RESTRICTED:
            # Check if user has approved access request
            has_access = any(
                req.user_id == user_id and 
                req.dataset_id == dataset_id and 
                req.status == "approved"
                for req in self.access_requests.values()
            )
            
            if not has_access:
                return {
                    "error": "access_denied",
                    "message": f"Dataset '{dataset_id}' requires access approval",
                    "action": "Submit access request via request_data_access()"
                }
        
        # Execute query through base API
        filters = filters or {}
        
        if query_type == "flows":
            result = self.base_api.query_research_data(
                query_type="raw_flows",
                dataset=dataset_id,
                limit=limit,
                anonymize=True  # Always anonymize for community access
            )
        elif query_type == "threats":
            result = self.base_api.query_research_data(
                query_type="threat_events",
                start_date=filters.get("start_date", "2026-01-01"),
                end_date=filters.get("end_date", "2026-12-31"),
                threat_type=filters.get("threat_type"),
                min_confidence=filters.get("min_confidence", 0.8),
                limit=limit
            )
        elif query_type == "statistics":
            result = self.base_api.query_research_data(
                query_type="statistics",
                stat_type=filters.get("stat_type", "threat_distribution")
            )
        else:
            return {
                "error": "invalid_query_type",
                "message": f"Query type '{query_type}' not supported"
            }
        
        # Increment quota
        quota = self._get_or_create_quota(user_id)
        quota.increment_usage()
        
        # Log access
        records_returned = result.get("records", 0) if isinstance(result, dict) else 0
        self._log_access(
            user_id=user_id,
            query_type=query_type,
            dataset_id=dataset_id,
            records_returned=records_returned,
            anonymized=True,
            ip_address=ip_address
        )
        
        # Add quota info to response
        if isinstance(result, dict):
            result["quota"] = {
                "queries_today": quota.queries_today,
                "daily_limit": quota.get_daily_limit(),
                "remaining": quota.get_daily_limit() - quota.queries_today
            }
        
        return result
    
    def request_data_access(
        self,
        user_id: str,
        dataset_id: str,
        purpose: str,
        institution: str
    ) -> Dict[str, Any]:
        """
        Request access to restricted dataset.
        
        Args:
            user_id: User identifier
            dataset_id: Dataset to request access to
            purpose: Research purpose
            institution: User's institution
            
        Returns:
            Access request confirmation
        """
        # Validate dataset exists
        if dataset_id not in self.datasets:
            return {
                "error": "dataset_not_found",
                "message": f"Dataset '{dataset_id}' does not exist"
            }
        
        dataset = self.datasets[dataset_id]
        
        # Check if dataset requires approval
        if dataset.access_level == DatasetAccessLevel.PUBLIC:
            return {
                "message": "Dataset is publicly accessible, no approval needed",
                "dataset_id": dataset_id
            }
        
        # Check for existing request
        existing_request = next(
            (req for req in self.access_requests.values()
             if req.user_id == user_id and req.dataset_id == dataset_id),
            None
        )
        
        if existing_request:
            return {
                "message": "Access request already exists",
                "request_id": existing_request.request_id,
                "status": existing_request.status
            }
        
        # Create new access request
        request_id = hashlib.sha256(
            f"{user_id}_{dataset_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        request = AccessRequest(
            request_id=request_id,
            user_id=user_id,
            dataset_id=dataset_id,
            purpose=purpose,
            institution=institution
        )
        
        self.access_requests[request_id] = request
        
        self.logger.info(f"Access request created: {request_id} for {user_id} -> {dataset_id}")
        
        return {
            "message": "Access request submitted successfully",
            "request_id": request_id,
            "status": "pending",
            "review_time": "Typically 2-5 business days"
        }
    
    def get_dataset_info(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dataset information and metadata.
        
        Args:
            dataset_id: Optional specific dataset ID
            
        Returns:
            Dataset information
        """
        if dataset_id:
            # Return specific dataset
            if dataset_id not in self.datasets:
                return {
                    "error": "dataset_not_found",
                    "message": f"Dataset '{dataset_id}' does not exist"
                }
            
            return self.datasets[dataset_id].to_dict()
        else:
            # Return all datasets
            return {
                "total_datasets": len(self.datasets),
                "datasets": [ds.to_dict() for ds in self.datasets.values()]
            }
    
    def get_user_quota(self, user_id: str) -> Dict[str, Any]:
        """
        Get user quota information.
        
        Args:
            user_id: User identifier
            
        Returns:
            Quota information
        """
        quota = self._get_or_create_quota(user_id)
        
        return {
            "user_id": user_id,
            "tier": quota.tier.value,
            "queries_today": quota.queries_today,
            "daily_limit": quota.get_daily_limit(),
            "remaining": quota.get_daily_limit() - quota.queries_today,
            "total_queries": quota.total_queries,
            "last_reset": quota.last_reset
        }
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            user_id: Optional filter by user
            limit: Maximum entries to return
            
        Returns:
            Audit log entries
        """
        logs = self.audit_log
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        # Return most recent entries
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [log.to_dict() for log in logs]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CommunityResearchAPI",
    "AccessTier",
    "DatasetAccessLevel",
    "UserQuota",
    "DatasetInfo",
    "AccessRequest",
    "AuditLogEntry"
]
