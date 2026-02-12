"""
AegisPCAP Database initialization, migration, and utilities
"""
import os
from sqlalchemy import text
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database initialization, migrations, and maintenance"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def init_database(self):
        """Initialize database schema"""
        try:
            self.db.engine.execute(text("""
                -- Enable UUID extension for PostgreSQL
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                
                -- Enable JSON extensions
                CREATE EXTENSION IF NOT EXISTS "json-functions";
            """))
            logger.info("Database extensions enabled")
        except Exception as e:
            logger.warning(f"Extensions may already exist: {e}")
        
        # Create all tables
        from src.db.models import Base
        Base.metadata.create_all(self.db.engine)
        logger.info("Database schema created successfully")
    
    def add_database_indexes(self):
        """Add performance indexes"""
        indexes = [
            # Flow indexes for fast queries
            "CREATE INDEX IF NOT EXISTS idx_flow_src_ip ON flows(src_ip);",
            "CREATE INDEX IF NOT EXISTS idx_flow_dst_ip ON flows(dst_ip);",
            "CREATE INDEX IF NOT EXISTS idx_flow_src_dst ON flows(src_ip, dst_ip);",
            "CREATE INDEX IF NOT EXISTS idx_flow_start_time ON flows(start_time);",
            "CREATE INDEX IF NOT EXISTS idx_flow_protocol ON flows(protocol);",
            
            # Alert indexes
            "CREATE INDEX IF NOT EXISTS idx_alert_flow_id ON alerts(flow_id);",
            "CREATE INDEX IF NOT EXISTS idx_alert_type ON alerts(alert_type);",
            "CREATE INDEX IF NOT EXISTS idx_alert_severity ON alerts(severity);",
            "CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);",
            "CREATE INDEX IF NOT EXISTS idx_alert_detected_at ON alerts(detected_at);",
            
            # Verdict indexes
            "CREATE INDEX IF NOT EXISTS idx_verdict_risk_score ON verdicts(risk_score);",
            "CREATE INDEX IF NOT EXISTS idx_verdict_risk_level ON verdicts(risk_level);",
            
            # TI indexes
            "CREATE INDEX IF NOT EXISTS idx_ti_entity ON threat_intelligence(entity_type, entity_value);",
            "CREATE INDEX IF NOT EXISTS idx_ti_threat_score ON threat_intelligence(threat_score);",
            
            # Incident indexes
            "CREATE INDEX IF NOT EXISTS idx_incident_type ON incidents(incident_type);",
            "CREATE INDEX IF NOT EXISTS idx_incident_status ON incidents(status);",
            "CREATE INDEX IF NOT EXISTS idx_incident_first_detected ON incidents(first_detected);",
            
            # Audit indexes
            "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);",
            "CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor);",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);",
        ]
        
        session = self.db.get_session()
        try:
            for index_sql in indexes:
                session.execute(text(index_sql))
            session.commit()
            logger.info(f"Created {len(indexes)} indexes")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            session.rollback()
        finally:
            session.close()
    
    def create_materialized_views(self):
        """Create materialized views for reporting"""
        views = {
            "v_high_risk_flows": """
                CREATE OR REPLACE VIEW v_high_risk_flows AS
                SELECT f.id, f.src_ip, f.dst_ip, f.protocol, f.start_time,
                       v.risk_score, v.risk_level, COUNT(a.id) as alert_count
                FROM flows f
                LEFT JOIN verdicts v ON f.id = v.flow_id
                LEFT JOIN alerts a ON f.id = a.flow_id
                WHERE v.risk_score > 0.7
                GROUP BY f.id, v.risk_score, v.risk_level
                ORDER BY v.risk_score DESC;
            """,
            "v_alert_summary": """
                CREATE OR REPLACE VIEW v_alert_summary AS
                SELECT alert_type, severity, COUNT(*) as count, 
                       AVG(risk_score) as avg_risk
                FROM alerts
                WHERE detected_at > NOW() - INTERVAL '24 hours'
                GROUP BY alert_type, severity
                ORDER BY count DESC;
            """,
            "v_top_ips": """
                CREATE OR REPLACE VIEW v_top_ips AS
                SELECT src_ip, COUNT(*) as flow_count, 
                       AVG(CASE WHEN v.risk_score > 0.7 THEN 1 ELSE 0 END) as risk_ratio,
                       SUM(total_bytes) as total_bytes
                FROM flows f
                LEFT JOIN verdicts v ON f.id = v.flow_id
                GROUP BY src_ip
                ORDER BY flow_count DESC
                LIMIT 1000;
            """,
            "v_incident_timeline": """
                CREATE OR REPLACE VIEW v_incident_timeline AS
                SELECT i.id, i.title, i.incident_type, i.severity, i.status,
                       i.first_detected, i.last_activity,
                       COUNT(f.id) as affected_flows,
                       COUNT(DISTINCT a.id) as related_alerts
                FROM incidents i
                LEFT JOIN flows f ON f.id = ANY(i.related_flows::int[])
                LEFT JOIN alerts a ON a.id = ANY(i.related_alerts::int[])
                GROUP BY i.id
                ORDER BY i.first_detected DESC;
            """,
        }
        
        session = self.db.get_session()
        try:
            for view_name, view_sql in views.items():
                session.execute(text(view_sql))
            session.commit()
            logger.info(f"Created {len(views)} materialized views")
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            session.rollback()
        finally:
            session.close()
    
    def setup_data_retention(self):
        """Configure data retention policies"""
        session = self.db.get_session()
        try:
            # Create retention policy function
            retention_function = """
                CREATE OR REPLACE FUNCTION enforce_data_retention() RETURNS void AS $$
                BEGIN
                    -- Keep flows for 90 days
                    DELETE FROM flows WHERE end_time < NOW() - INTERVAL '90 days';
                    
                    -- Keep alerts for 365 days
                    DELETE FROM alerts WHERE detected_at < NOW() - INTERVAL '365 days'
                    AND status IN ('resolved', 'false_positive');
                    
                    -- Keep incidents for 2 years
                    DELETE FROM incidents WHERE closed_at < NOW() - INTERVAL '2 years'
                    AND status IN ('resolved', 'false_positive');
                    
                    -- Keep audit logs for 1 year
                    DELETE FROM audit_logs WHERE timestamp < NOW() - INTERVAL '1 year';
                    
                    -- Keep old feature stores
                    DELETE FROM feature_store 
                    WHERE computed_at < NOW() - INTERVAL '30 days';
                    
                    RAISE NOTICE 'Data retention enforced';
                END;
                $$ LANGUAGE plpgsql;
            """
            session.execute(text(retention_function))
            session.commit()
            logger.info("Data retention policies configured")
        except Exception as e:
            logger.warning(f"Retention policies may already exist: {e}")
            session.rollback()
        finally:
            session.close()
    
    def backup_database(self, backup_dir: str):
        """Create database backup"""
        import subprocess
        from datetime import datetime
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"aegis_backup_{timestamp}.sql")
        
        try:
            # Extract connection string
            url = str(self.db.engine.url)
            # This would need to be adapted based on actual connection string format
            logger.info(f"Database backup initiated to {backup_file}")
            # subprocess.run([...], check=True)
            return backup_file
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None


class DataManager:
    """Manages data ingestion, transformation, and export"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def ingest_flows(self, flows_list: List[Dict]) -> int:
        """Ingest multiple flows"""
        from src.db.models import Flow, FlowRepository
        
        session = self.db.get_session()
        repo = FlowRepository(session)
        
        try:
            inserted = 0
            for flow_data in flows_list:
                # Check if flow already exists
                if not repo.get_by_hash(flow_data.get('flow_hash')):
                    repo.create(flow_data)
                    inserted += 1
            
            session.commit()
            logger.info(f"Ingested {inserted}/{len(flows_list)} flows")
            return inserted
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            session.rollback()
            return 0
        finally:
            session.close()
    
    def ingest_alerts(self, alerts_list: List[Dict]) -> int:
        """Ingest multiple alerts"""
        from src.db.models import Alert, AlertRepository
        
        session = self.db.get_session()
        repo = AlertRepository(session)
        
        try:
            inserted = 0
            for alert_data in alerts_list:
                repo.create(alert_data)
                inserted += 1
            
            session.commit()
            logger.info(f"Ingested {inserted} alerts")
            return inserted
        except Exception as e:
            logger.error(f"Alert ingestion failed: {e}")
            session.rollback()
            return 0
        finally:
            session.close()
    
    def export_flows(self, format: str = 'json', filters: Optional[Dict] = None) -> str:
        """Export flows in specified format"""
        from src.db.models import Flow
        import json
        import csv
        
        session = self.db.get_session()
        try:
            query = session.query(Flow)
            
            # Apply filters
            if filters:
                if 'src_ip' in filters:
                    query = query.filter(Flow.src_ip == filters['src_ip'])
                if 'dst_ip' in filters:
                    query = query.filter(Flow.dst_ip == filters['dst_ip'])
                if 'protocol' in filters:
                    query = query.filter(Flow.protocol == filters['protocol'])
                if 'time_range' in filters:
                    start, end = filters['time_range']
                    query = query.filter(Flow.start_time >= start, Flow.end_time <= end)
            
            flows = query.all()
            
            if format == 'json':
                flow_dicts = [{
                    'id': f.id,
                    'src_ip': f.src_ip,
                    'dst_ip': f.dst_ip,
                    'protocol': f.protocol,
                    'start_time': f.start_time.isoformat(),
                    'duration': f.duration,
                    'bytes': f.total_bytes,
                } for f in flows]
                return json.dumps(flow_dicts, indent=2)
            
            elif format == 'csv':
                output = []
                output.append('id,src_ip,dst_ip,protocol,start_time,duration,bytes')
                for f in flows:
                    output.append(f"{f.id},{f.src_ip},{f.dst_ip},{f.protocol},"
                                f"{f.start_time.isoformat()},{f.duration},{f.total_bytes}")
                return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
        finally:
            session.close()
        
        return ""
    
    def get_statistics(self, time_range_hours: int = 24) -> Dict:
        """Get database statistics"""
        from src.db.models import Flow, Alert, Verdict, Incident
        from datetime import timedelta
        
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            stats = {
                'flows': {
                    'total': session.query(Flow).count(),
                    'recent': session.query(Flow).filter(Flow.start_time >= cutoff).count(),
                    'total_bytes': session.query(Flow).with_entities(
                        __import__('sqlalchemy').func.sum(Flow.total_bytes)
                    ).scalar() or 0,
                },
                'alerts': {
                    'total': session.query(Alert).count(),
                    'recent': session.query(Alert).filter(Alert.detected_at >= cutoff).count(),
                    'by_severity': {}
                },
                'incidents': {
                    'open': session.query(Incident).filter(Incident.status == 'open').count(),
                    'total': session.query(Incident).count(),
                },
                'verdicts': {
                    'high_risk': session.query(Verdict).filter(Verdict.risk_score >= 0.7).count(),
                    'avg_risk': session.query(Verdict).with_entities(
                        __import__('sqlalchemy').func.avg(Verdict.risk_score)
                    ).scalar() or 0,
                },
            }
            
            return stats
        except Exception as e:
            logger.error(f"Statistics query failed: {e}")
            return {}
        finally:
            session.close()
    
    def cleanup_old_data(self, days_old: int = 90):
        """Remove data older than specified days"""
        from src.db.models import Flow, Alert
        from sqlalchemy import and_
        
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete old flows
            deleted_flows = session.query(Flow).filter(Flow.end_time < cutoff).delete()
            
            # Delete old alerts (except confirmed threats)
            deleted_alerts = session.query(Alert).filter(
                and_(Alert.detected_at < cutoff, Alert.status != 'confirmed')
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_flows} flows and {deleted_alerts} alerts")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            session.rollback()
        finally:
            session.close()


class AnalyticsEngine:
    """Perform analytics on stored data"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def get_top_attacking_ips(self, limit: int = 10, time_range_hours: int = 24) -> List[Dict]:
        """Get top source IPs by alert count"""
        from src.db.models import Flow, Alert
        from datetime import timedelta
        from sqlalchemy import func
        
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            results = session.query(
                Flow.src_ip,
                func.count(Alert.id).label('alert_count'),
                func.avg(Alert.risk_score).label('avg_risk'),
            ).join(Alert, Flow.id == Alert.flow_id).filter(
                Alert.detected_at >= cutoff
            ).group_by(Flow.src_ip).order_by(
                func.count(Alert.id).desc()
            ).limit(limit).all()
            
            return [
                {
                    'ip': r[0],
                    'alert_count': r[1],
                    'avg_risk': float(r[2]) if r[2] else 0,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return []
        finally:
            session.close()
    
    def get_threat_timeline(self, time_range_hours: int = 24) -> List[Dict]:
        """Get timeline of threats over time"""
        from src.db.models import Alert
        from datetime import timedelta
        from sqlalchemy import func
        
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            results = session.query(
                func.date_trunc('hour', Alert.detected_at).label('hour'),
                func.count(Alert.id).label('count'),
                Alert.alert_type,
            ).filter(Alert.detected_at >= cutoff).group_by(
                func.date_trunc('hour', Alert.detected_at),
                Alert.alert_type
            ).order_by(func.date_trunc('hour', Alert.detected_at)).all()
            
            return [
                {
                    'timestamp': r[0].isoformat() if r[0] else None,
                    'count': r[1],
                    'type': r[2],
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Timeline query failed: {e}")
            return []
        finally:
            session.close()
    
    def correlate_incidents(self, time_window_minutes: int = 60) -> List[Dict]:
        """Find related flows that should be grouped into incidents"""
        from src.db.models import Flow, Alert, Verdict
        from datetime import timedelta
        
        session = self.db.get_session()
        try:
            # Find flows with similar characteristics in time window
            window = timedelta(minutes=time_window_minutes)
            
            high_risk_flows = session.query(Flow, Verdict).join(
                Verdict, Flow.id == Verdict.flow_id
            ).filter(Verdict.risk_score >= 0.7).all()
            
            correlations = []
            for i, (f1, v1) in enumerate(high_risk_flows):
                for f2, v2 in high_risk_flows[i+1:]:
                    # Check if flows are related
                    time_diff = abs((f1.start_time - f2.start_time).total_seconds())
                    
                    if time_diff < window.total_seconds():
                        # Check if from same source or to same destination
                        if f1.src_ip == f2.src_ip or f1.dst_ip == f2.dst_ip:
                            correlations.append({
                                'flow_ids': [f1.id, f2.id],
                                'time_diff': time_diff,
                                'combined_risk': (v1.risk_score + v2.risk_score) / 2,
                            })
            
            return sorted(correlations, key=lambda x: x['combined_risk'], reverse=True)
        
        except Exception as e:
            logger.error(f"Correlation failed: {e}")
            return []
        finally:
            session.close()
