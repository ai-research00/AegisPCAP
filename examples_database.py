#!/usr/bin/env python3
"""
AegisPCAP Phase 5: Database Layer Quick Start
Demonstrates initialization and usage of persistence layer
"""

import logging
from datetime import datetime, timedelta
from src.db.persistence import get_persistence_layer, initialize_persistence
from src.db.config import DatabaseConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_initialize():
    """Example 1: Initialize persistence layer"""
    logger.info("=" * 60)
    logger.info("Example 1: Initialize Persistence Layer")
    logger.info("=" * 60)
    
    # Initialize globally
    if initialize_persistence():
        logger.info("✅ Persistence layer initialized")
    else:
        logger.error("❌ Failed to initialize")
    
    # Get instance
    persistence = get_persistence_layer()
    
    # Test connectivity
    status = persistence.test_connectivity()
    logger.info(f"Database connectivity: {status['database']}")
    logger.info(f"Redis connectivity: {status['redis']}")


def example_2_save_flow():
    """Example 2: Save a network flow"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Save Network Flow")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Create flow data
    flow_data = {
        'flow_hash': 'abc123def456',
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.50',
        'src_port': 54321,
        'dst_port': 443,
        'protocol': 'TCP',
        'start_time': datetime.utcnow() - timedelta(hours=1),
        'end_time': datetime.utcnow(),
        'duration': 3600.0,
        'packet_count': 1000,
        'total_bytes': 1048576,
        'fwd_packets': 500,
        'bwd_packets': 500,
        'fwd_bytes': 524288,
        'bwd_bytes': 524288,
        'src_country': 'US',
        'src_city': 'New York',
        'dst_country': 'US',
        'dst_city': 'San Francisco',
        'dns_queries': ['google.com', 'api.google.com'],
        'tls_snis': ['google.com'],
    }
    
    # Save flow
    flow_id = persistence.save_flow(flow_data)
    
    if flow_id:
        logger.info(f"✅ Flow saved with ID: {flow_id}")
        
        # Retrieve flow
        retrieved = persistence.get_flow(flow_id)
        logger.info(f"Retrieved flow: {retrieved['src_ip']} → {retrieved['dst_ip']}")
    else:
        logger.error("❌ Failed to save flow")


def example_3_save_features():
    """Example 3: Save and cache extracted features"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Cache Extracted Features")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Create flow first
    flow_data = {
        'flow_hash': 'feature_test_001',
        'src_ip': '192.168.1.200',
        'dst_ip': '10.0.0.100',
        'protocol': 'TCP',
        'start_time': datetime.utcnow(),
        'end_time': datetime.utcnow(),
    }
    flow_id = persistence.save_flow(flow_data)
    
    # Create features
    features = {
        'pkt_size_mean': 512.5,
        'pkt_size_std': 128.3,
        'mean_iat': 0.1,
        'std_iat': 0.02,
        'dns_entropy': 4.5,
        'dns_dga_score': 0.15,
        'tls_c2_score': 0.05,
        'beaconing_score': 0.1,
    }
    
    # Cache features
    if persistence.cache_flow_features(flow_id, features):
        logger.info(f"✅ Features cached for flow {flow_id}")
        
        # Retrieve features
        cached_features = persistence.get_cached_features(flow_id)
        logger.info(f"Cached features: {len(cached_features) if cached_features else 0} items")
    else:
        logger.info("⚠️ Feature caching unavailable (Redis not running)")


def example_4_save_alert():
    """Example 4: Save security alert"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Save Security Alert")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Create flow first
    flow_data = {
        'flow_hash': 'alert_test_001',
        'src_ip': '192.168.1.50',
        'dst_ip': '10.0.0.200',
        'protocol': 'TCP',
        'start_time': datetime.utcnow(),
        'end_time': datetime.utcnow(),
    }
    flow_id = persistence.save_flow(flow_data)
    
    # Create alert
    alert_data = {
        'flow_id': flow_id,
        'alert_type': 'C2',
        'severity': 'HIGH',
        'risk_score': 0.85,
        'confidence': 0.9,
        'detection_source': 'ensemble_ml',
        'evidence': [
            'High periodicity detected (0.92)',
            'TLS certificate reuse (0.87)',
            'Uncommon destination port (443)',
        ],
        'mitre_tactics': ['Command and Control', 'Defense Evasion'],
        'mitre_techniques': ['Application Layer Protocol', 'Encrypted Channel'],
        'status': 'new',
    }
    
    # Save alert
    alert_id = persistence.save_alert(alert_data)
    
    if alert_id:
        logger.info(f"✅ Alert saved with ID: {alert_id}")
        logger.info(f"Type: {alert_data['alert_type']}, Severity: {alert_data['severity']}")
    else:
        logger.error("❌ Failed to save alert")


def example_5_query_alerts():
    """Example 5: Query alerts"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Query Alerts")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Get alerts
    alerts = persistence.get_alerts(limit=5)
    
    logger.info(f"Total alerts retrieved: {len(alerts)}")
    for alert in alerts[:3]:
        logger.info(f"  Alert {alert['id']}: {alert['type']} (Risk: {alert['risk_score']:.2f})")


def example_6_statistics():
    """Example 6: Get system statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: System Statistics")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Get stats
    stats = persistence.get_statistics(time_range_hours=24)
    
    logger.info(f"Last 24 hours statistics:")
    logger.info(f"  Total flows: {stats.get('flows', {}).get('total', 0)}")
    logger.info(f"  Recent flows: {stats.get('flows', {}).get('recent', 0)}")
    logger.info(f"  Total bytes: {stats.get('flows', {}).get('total_bytes', 0)}")
    logger.info(f"  Recent alerts: {stats.get('alerts', {}).get('recent', 0)}")
    logger.info(f"  Open incidents: {stats.get('incidents', {}).get('open', 0)}")


def example_7_analytics():
    """Example 7: Threat analytics"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: Threat Analytics")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Get top IPs
    top_ips = persistence.get_top_attacking_ips(limit=5, hours=24)
    
    logger.info(f"Top attacking IPs (24h):")
    for ip_info in top_ips:
        logger.info(f"  {ip_info['ip']}: {ip_info['alert_count']} alerts, Risk: {ip_info['avg_risk']:.2f}")
    
    # Get threat timeline
    timeline = persistence.get_threat_timeline(hours=24)
    
    logger.info(f"Threat timeline (24h): {len(timeline)} data points")
    if timeline:
        logger.info(f"  Sample: {timeline[0]['timestamp']} - {timeline[0]['count']} {timeline[0]['type']} alerts")


def example_8_pool_status():
    """Example 8: Monitor connection pool"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 8: Connection Pool Status")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Get pool status
    pool_status = persistence.get_pool_status()
    
    logger.info(f"Connection pool status:")
    for key, value in pool_status.items():
        logger.info(f"  {key}: {value}")
    
    # Get cache stats
    cache_stats = persistence.get_cache_stats()
    
    logger.info(f"Redis cache stats:")
    for key, value in cache_stats.items():
        logger.info(f"  {key}: {value}")


def example_9_export():
    """Example 9: Export data"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 9: Export Data")
    logger.info("=" * 60)
    
    persistence = get_persistence_layer()
    
    # Export flows as JSON
    data = persistence.export_data(format='json', filters={
        'src_ip': '192.168.1.100'
    })
    
    if data:
        logger.info(f"✅ Exported {len(data)} bytes of data")
        # Show first 200 chars
        logger.info(f"Sample: {data[:200]}...")
    else:
        logger.info("⚠️ No data to export")


def example_10_configuration():
    """Example 10: View configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 10: Database Configuration")
    logger.info("=" * 60)
    
    # Show current configuration
    config = DatabaseConfig()
    
    logger.info(f"Primary database type: {config.db_type.value}")
    logger.info(f"PostgreSQL: {config.postgresql.host}:{config.postgresql.port}/{config.postgresql.database}")
    logger.info(f"Redis: {config.redis.host}:{config.redis.port}/{config.redis.db}")
    logger.info(f"Elasticsearch: {config.elasticsearch.host}:{config.elasticsearch.port}")
    
    # Connection URLs
    logger.info(f"Primary DB URL: {config.get_primary_db_url()}")
    logger.info(f"Cache URL: {config.get_cache_url()}")


def main():
    """Run all examples"""
    logger.info("🚀 AegisPCAP Phase 5 - Database Layer Examples\n")
    
    try:
        # Initialize persistence layer first
        example_1_initialize()
        
        # Run examples (skip if init failed)
        example_2_save_flow()
        example_3_save_features()
        example_4_save_alert()
        example_5_query_alerts()
        example_6_statistics()
        example_7_analytics()
        example_8_pool_status()
        example_9_export()
        example_10_configuration()
        
        logger.info("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during examples: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("\n🔄 Closing persistence layer...")
        persistence = get_persistence_layer()
        persistence.close()
        logger.info("✅ Cleanup complete")


if __name__ == '__main__':
    main()
