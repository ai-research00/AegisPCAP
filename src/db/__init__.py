"""
AegisPCAP Database module - exports all database components
"""
from src.db.models import (
    Base,
    Flow,
    FlowFeatures,
    Alert,
    Verdict,
    ThreatIntelligence,
    Incident,
    AuditLog,
    ModelMetadata,
    FeatureStore,
    DatabaseSession,
    FlowRepository,
    AlertRepository,
    IncidentRepository,
)

from src.db.manager import (
    DatabaseManager,
    DataManager,
    AnalyticsEngine,
)

from src.db.cache import (
    RedisCache,
    FeatureStore as FeatureStoreCache,
    SessionCache,
)

from src.db.connection import (
    DatabaseConnection,
    ContextManager,
    DatabaseProvider,
    DatabaseMiddleware,
    QueryOptimizer,
    DatabaseException,
    ConnectionException,
    QueryException,
)

__all__ = [
    # Models
    'Base',
    'Flow',
    'FlowFeatures',
    'Alert',
    'Verdict',
    'ThreatIntelligence',
    'Incident',
    'AuditLog',
    'ModelMetadata',
    'FeatureStore',
    'DatabaseSession',
    
    # Repositories
    'FlowRepository',
    'AlertRepository',
    'IncidentRepository',
    
    # Managers
    'DatabaseManager',
    'DataManager',
    'AnalyticsEngine',
    
    # Caching
    'RedisCache',
    'FeatureStoreCache',
    'SessionCache',
    
    # Connection
    'DatabaseConnection',
    'ContextManager',
    'DatabaseProvider',
    'DatabaseMiddleware',
    'QueryOptimizer',
    
    # Exceptions
    'DatabaseException',
    'ConnectionException',
    'QueryException',
]
