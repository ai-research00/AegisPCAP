"""
AegisPCAP Phase 5: Data Persistence - Integration module
Orchestrates all database, caching, and persistence components
"""
import logging
from typing import Optional, Dict, List
from datetime import datetime

from src.db.connection import DatabaseConnection, DatabaseProvider, ContextManager
from src.db.models import DatabaseSession, Flow, Alert, Verdict, Incident
from src.db.manager import DatabaseManager, DataManager, AnalyticsEngine
from src.db.cache import RedisCache, FeatureStore as FeatureStoreCache, SessionCache
from src.db.config import DatabaseConfig

logger = logging.getLogger(__name__)


class PersistenceLayer:
    """
    Unified persistence layer that orchestrates:
    - PostgreSQL for structured data (flows, alerts, incidents)
    - Redis for caching and feature store
    - Elasticsearch for search/analytics
    - MongoDB for unstructured data
    """
    
    def __init__(self):
        self.db_connection: Optional[DatabaseConnection] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.data_manager: Optional[DataManager] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None
        
        self.redis_cache: Optional[RedisCache] = None
        self.feature_store: Optional[FeatureStoreCache] = None
        self.session_cache: Optional[SessionCache] = None
    
    def initialize(self) -> bool:
        """Initialize all persistence components"""
        try:
            # Initialize primary database
            db_url = DatabaseConfig.get_primary_db_url()
            self.db_connection = DatabaseConnection(db_url)
            
            if not self.db_connection.initialize():
                logger.error("Failed to initialize database connection")
                return False
            
            # Initialize database schema
            self.db_manager = DatabaseManager(self.db_connection)
            self.db_manager.init_database()
            self.db_manager.add_database_indexes()
            self.db_manager.create_materialized_views()
            self.db_manager.setup_data_retention()
            
            # Initialize data managers
            session = self.db_connection.get_session()
            self.data_manager = DataManager(self.db_connection)
            self.analytics_engine = AnalyticsEngine(self.db_connection)
            session.close()
            
            # Initialize Redis caching
            redis_url = DatabaseConfig.get_cache_url()
            self.redis_cache = RedisCache(
                host=DatabaseConfig.redis.host,
                port=DatabaseConfig.redis.port,
                db=DatabaseConfig.redis.db,
                password=DatabaseConfig.redis.password
            )
            
            if not self.redis_cache.is_available():
                logger.warning("Redis not available - caching disabled")
            else:
                # Initialize feature store and session cache
                self.feature_store = FeatureStoreCache(self.redis_cache)
                self.session_cache = SessionCache(self.redis_cache)
                logger.info("Redis caching initialized")
            
            logger.info("✅ Persistence layer fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Persistence layer initialization failed: {e}")
            return False
    
    def test_connectivity(self) -> Dict[str, bool]:
        """Test all persistence layer connections"""
        results = {
            'database': False,
            'redis': False,
            'elasticsearch': False,
        }
        
        try:
            results['database'] = self.db_connection.test_connection()
        except:
            results['database'] = False
        
        try:
            results['redis'] = self.redis_cache.is_available()
        except:
            results['redis'] = False
        
        return results
    
    # Flow persistence
    def save_flow(self, flow_data: Dict) -> Optional[int]:
        """Save flow to database and cache"""
        try:
            # Save to database
            session = self.db_connection.get_session()
            flow = Flow(**flow_data)
            session.add(flow)
            session.commit()
            flow_id = flow.id
            session.close()
            
            # Cache flow
            if self.redis_cache:
                self.redis_cache.cache_flow(flow_id, flow_data)
            
            logger.debug(f"Flow {flow_id} saved")
            return flow_id
            
        except Exception as e:
            logger.error(f"Failed to save flow: {e}")
            return None
    
    def get_flow(self, flow_id: int) -> Optional[Dict]:
        """Retrieve flow with cache fallback"""
        try:
            # Try cache first
            if self.redis_cache:
                cached = self.redis_cache.get_flow(flow_id)
                if cached:
                    return cached
            
            # Fall back to database
            session = self.db_connection.get_session()
            flow = session.query(Flow).filter(Flow.id == flow_id).first()
            session.close()
            
            if flow:
                flow_dict = {
                    'id': flow.id,
                    'src_ip': flow.src_ip,
                    'dst_ip': flow.dst_ip,
                    'protocol': flow.protocol,
                    'start_time': flow.start_time.isoformat(),
                    'duration': flow.duration,
                }
                
                # Cache for future
                if self.redis_cache:
                    self.redis_cache.cache_flow(flow_id, flow_dict)
                
                return flow_dict
            
        except Exception as e:
            logger.error(f"Failed to retrieve flow {flow_id}: {e}")
        
        return None
    
    # Alert persistence
    def save_alert(self, alert_data: Dict) -> Optional[int]:
        """Save alert to database and cache"""
        try:
            session = self.db_connection.get_session()
            alert = Alert(**alert_data)
            session.add(alert)
            session.commit()
            alert_id = alert.id
            session.close()
            
            if self.redis_cache:
                self.redis_cache.cache_alert(alert_id, alert_data)
            
            logger.info(f"Alert {alert_id} saved - Type: {alert_data.get('alert_type')}, Severity: {alert_data.get('severity')}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return None
    
    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        try:
            session = self.db_connection.get_session()
            query = session.query(Alert)
            
            if severity:
                query = query.filter(Alert.severity == severity)
            
            alerts = query.order_by(Alert.detected_at.desc()).limit(limit).all()
            session.close()
            
            return [
                {
                    'id': a.id,
                    'type': a.alert_type,
                    'severity': a.severity,
                    'risk_score': a.risk_score,
                    'detected_at': a.detected_at.isoformat(),
                    'status': a.status,
                }
                for a in alerts
            ]
            
        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            return []
    
    # Incident persistence
    def save_incident(self, incident_data: Dict) -> Optional[int]:
        """Save incident to database"""
        try:
            session = self.db_connection.get_session()
            incident = Incident(**incident_data)
            session.add(incident)
            session.commit()
            incident_id = incident.id
            session.close()
            
            if self.redis_cache:
                self.redis_cache.cache_incident(incident_id, incident_data)
            
            logger.warning(f"Incident {incident_id} created - Type: {incident_data.get('incident_type')}, Severity: {incident_data.get('severity')}")
            return incident_id
            
        except Exception as e:
            logger.error(f"Failed to save incident: {e}")
            return None
    
    # Feature store operations
    def cache_flow_features(self, flow_id: int, features: Dict) -> bool:
        """Cache extracted features for flow"""
        if not self.feature_store:
            return False
        
        try:
            self.feature_store.store_flow_features(flow_id, features)
            return True
        except Exception as e:
            logger.error(f"Failed to cache features: {e}")
            return False
    
    def get_cached_features(self, flow_id: int) -> Optional[Dict]:
        """Retrieve cached features"""
        if not self.feature_store:
            return None
        
        try:
            return self.feature_store.get_flow_features(flow_id)
        except Exception as e:
            logger.error(f"Failed to retrieve cached features: {e}")
            return None
    
    # Analytics and reporting
    def get_statistics(self, time_range_hours: int = 24) -> Dict:
        """Get system statistics"""
        try:
            return self.data_manager.get_statistics(time_range_hours)
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def get_threat_timeline(self, hours: int = 24) -> List[Dict]:
        """Get threat detection timeline"""
        try:
            return self.analytics_engine.get_threat_timeline(hours)
        except Exception as e:
            logger.error(f"Failed to get threat timeline: {e}")
            return []
    
    def get_top_attacking_ips(self, limit: int = 10, hours: int = 24) -> List[Dict]:
        """Get top source IPs by threat"""
        try:
            return self.analytics_engine.get_top_attacking_ips(limit, hours)
        except Exception as e:
            logger.error(f"Failed to get top IPs: {e}")
            return []
    
    def correlate_incidents(self, time_window_minutes: int = 60) -> List[Dict]:
        """Find related flows for incident grouping"""
        try:
            return self.analytics_engine.correlate_incidents(time_window_minutes)
        except Exception as e:
            logger.error(f"Failed to correlate incidents: {e}")
            return []
    
    # Data management
    def cleanup_old_data(self, days_old: int = 90):
        """Remove old data according to retention policy"""
        try:
            self.data_manager.cleanup_old_data(days_old)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def export_data(self, format: str = 'json', filters: Optional[Dict] = None) -> str:
        """Export data in specified format"""
        try:
            return self.data_manager.export_flows(format, filters)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""
    
    # Connection management
    def get_pool_status(self) -> Dict:
        """Get connection pool status"""
        if self.db_connection:
            return self.db_connection.get_pool_status()
        return {}
    
    def get_cache_stats(self) -> Dict:
        """Get Redis cache statistics"""
        if self.redis_cache:
            return self.redis_cache.get_stats()
        return {}
    
    def close(self):
        """Close all connections"""
        try:
            if self.db_connection:
                self.db_connection.dispose()
            if self.redis_cache:
                self.redis_cache.redis.close()
            logger.info("Persistence layer closed")
        except Exception as e:
            logger.error(f"Close failed: {e}")


# Global persistence instance
_persistence_layer: Optional[PersistenceLayer] = None


def get_persistence_layer() -> PersistenceLayer:
    """Get or create global persistence layer"""
    global _persistence_layer
    
    if _persistence_layer is None:
        _persistence_layer = PersistenceLayer()
        _persistence_layer.initialize()
    
    return _persistence_layer


def initialize_persistence() -> bool:
    """Initialize global persistence layer"""
    layer = get_persistence_layer()
    return layer.db_connection is not None
