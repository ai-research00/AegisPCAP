"""
AegisPCAP Redis-based caching and feature store
"""
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pickle

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching layer for high-performance access"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: Optional[str] = None):
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # Handle binary data
                socket_keepalive=True,
                socket_keepalive_options=True,
            )
            # Test connection
            self.redis.ping()
            logger.info(f"Redis connected to {host}:{port}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        try:
            return self.redis is not None and self.redis.ping()
        except:
            return False
    
    # Flow caching
    def cache_flow(self, flow_id: int, flow_data: Dict, ttl: int = 3600):
        """Cache flow data"""
        if not self.is_available():
            return False
        
        try:
            key = f"flow:{flow_id}"
            self.redis.setex(key, ttl, json.dumps(flow_data))
            return True
        except Exception as e:
            logger.error(f"Flow cache failed: {e}")
            return False
    
    def get_flow(self, flow_id: int) -> Optional[Dict]:
        """Retrieve cached flow"""
        if not self.is_available():
            return None
        
        try:
            key = f"flow:{flow_id}"
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Flow cache retrieval failed: {e}")
            return None
    
    # Alert caching
    def cache_alert(self, alert_id: int, alert_data: Dict, ttl: int = 7200):
        """Cache alert data"""
        if not self.is_available():
            return False
        
        try:
            key = f"alert:{alert_id}"
            self.redis.setex(key, ttl, json.dumps(alert_data))
            return True
        except Exception as e:
            logger.error(f"Alert cache failed: {e}")
            return False
    
    def get_alert(self, alert_id: int) -> Optional[Dict]:
        """Retrieve cached alert"""
        if not self.is_available():
            return None
        
        try:
            key = f"alert:{alert_id}"
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Alert cache retrieval failed: {e}")
            return None
    
    # Incident caching
    def cache_incident(self, incident_id: int, incident_data: Dict, ttl: int = 86400):
        """Cache incident data"""
        if not self.is_available():
            return False
        
        try:
            key = f"incident:{incident_id}"
            self.redis.setex(key, ttl, json.dumps(incident_data))
            return True
        except Exception as e:
            logger.error(f"Incident cache failed: {e}")
            return False
    
    def get_incident(self, incident_id: int) -> Optional[Dict]:
        """Retrieve cached incident"""
        if not self.is_available():
            return None
        
        try:
            key = f"incident:{incident_id}"
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Incident cache retrieval failed: {e}")
            return None
    
    # Rate limiting
    def rate_limit(self, client_id: str, max_requests: int = 100, window: int = 60) -> bool:
        """Implement rate limiting"""
        if not self.is_available():
            return True  # Allow if Redis unavailable
        
        try:
            key = f"ratelimit:{client_id}"
            current = self.redis.incr(key)
            
            if current == 1:
                self.redis.expire(key, window)
            
            return current <= max_requests
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True
    
    # Bloom filter for deduplication
    def add_to_bloom_filter(self, filter_name: str, item: str):
        """Add item to bloom filter"""
        if not self.is_available():
            return False
        
        try:
            # Using Redis Set as simple bloom filter alternative
            key = f"bloom:{filter_name}"
            self.redis.sadd(key, item)
            self.redis.expire(key, 86400)  # 24 hours
            return True
        except Exception as e:
            logger.error(f"Bloom filter add failed: {e}")
            return False
    
    def check_bloom_filter(self, filter_name: str, item: str) -> bool:
        """Check if item in bloom filter"""
        if not self.is_available():
            return False
        
        try:
            key = f"bloom:{filter_name}"
            return self.redis.sismember(key, item)
        except Exception as e:
            logger.error(f"Bloom filter check failed: {e}")
            return False
    
    # Counters and metrics
    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment counter"""
        if not self.is_available():
            return None
        
        try:
            key = f"counter:{counter_name}"
            return self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Counter increment failed: {e}")
            return None
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        if not self.is_available():
            return 0
        
        try:
            key = f"counter:{counter_name}"
            val = self.redis.get(key)
            return int(val) if val else 0
        except Exception as e:
            logger.error(f"Counter retrieval failed: {e}")
            return 0
    
    # Queue operations
    def enqueue(self, queue_name: str, item: Dict):
        """Add item to queue"""
        if not self.is_available():
            return False
        
        try:
            key = f"queue:{queue_name}"
            self.redis.rpush(key, json.dumps(item))
            return True
        except Exception as e:
            logger.error(f"Queue enqueue failed: {e}")
            return False
    
    def dequeue(self, queue_name: str) -> Optional[Dict]:
        """Get item from queue"""
        if not self.is_available():
            return None
        
        try:
            key = f"queue:{queue_name}"
            item = self.redis.lpop(key)
            return json.loads(item) if item else None
        except Exception as e:
            logger.error(f"Queue dequeue failed: {e}")
            return None
    
    def queue_length(self, queue_name: str) -> int:
        """Get queue length"""
        if not self.is_available():
            return 0
        
        try:
            key = f"queue:{queue_name}"
            return self.redis.llen(key)
        except Exception as e:
            logger.error(f"Queue length check failed: {e}")
            return 0
    
    # Clear cache
    def clear_cache(self, pattern: str = "*"):
        """Clear cache by pattern"""
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get Redis statistics"""
        if not self.is_available():
            return {}
        
        try:
            info = self.redis.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {}


class FeatureStore:
    """Redis-backed feature store for ML feature caching"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
    
    def store_flow_features(self, flow_id: int, features: Dict, ttl: int = 3600):
        """Store flow features in feature store"""
        if not self.cache.is_available():
            return False
        
        try:
            key = f"features:flow:{flow_id}"
            features_with_timestamp = {
                **features,
                'cached_at': datetime.utcnow().isoformat(),
            }
            self.cache.redis.setex(key, ttl, json.dumps(features_with_timestamp))
            
            # Add to feature set index
            self.cache.redis.sadd(f"features:flow:ids", flow_id)
            return True
        except Exception as e:
            logger.error(f"Feature store failed: {e}")
            return False
    
    def get_flow_features(self, flow_id: int) -> Optional[Dict]:
        """Retrieve cached flow features"""
        if not self.cache.is_available():
            return None
        
        try:
            key = f"features:flow:{flow_id}"
            data = self.cache.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Feature retrieval failed: {e}")
            return None
    
    def store_entity_features(self, entity_type: str, entity_value: str, features: Dict, ttl: int = 86400):
        """Store pre-computed features for entities (IPs, domains)"""
        if not self.cache.is_available():
            return False
        
        try:
            key = f"features:{entity_type}:{entity_value}"
            features_with_timestamp = {
                **features,
                'cached_at': datetime.utcnow().isoformat(),
            }
            self.cache.redis.setex(key, ttl, json.dumps(features_with_timestamp))
            return True
        except Exception as e:
            logger.error(f"Entity feature store failed: {e}")
            return False
    
    def get_entity_features(self, entity_type: str, entity_value: str) -> Optional[Dict]:
        """Retrieve pre-computed entity features"""
        if not self.cache.is_available():
            return None
        
        try:
            key = f"features:{entity_type}:{entity_value}"
            data = self.cache.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Entity feature retrieval failed: {e}")
            return None
    
    def get_similar_flows(self, flow_features: Dict, limit: int = 10) -> List[int]:
        """Find similar flows based on cached features"""
        if not self.cache.is_available():
            return []
        
        try:
            # Simple similarity based on protocol, port patterns
            protocol = flow_features.get('protocol')
            port = flow_features.get('src_port', 0)
            
            # Search for similar flows in cache
            pattern = f"features:flow:*"
            keys = self.cache.redis.keys(pattern)
            
            similar = []
            for key in keys[:1000]:  # Limit search
                cached = self.cache.redis.get(key)
                if cached:
                    cf = json.loads(cached)
                    if cf.get('protocol') == protocol:
                        similar.append(int(key.split(':')[-1]))
            
            return similar[:limit]
        except Exception as e:
            logger.error(f"Similar flows search failed: {e}")
            return []
    
    def get_feature_stats(self) -> Dict:
        """Get feature store statistics"""
        if not self.cache.is_available():
            return {}
        
        try:
            keys = self.cache.redis.keys("features:flow:*")
            return {
                'cached_flows': len(keys),
                'cache_size_bytes': sum(self.cache.redis.memory_usage(k) or 0 for k in keys[:100]),
            }
        except Exception as e:
            logger.error(f"Feature stats failed: {e}")
            return {}


class SessionCache:
    """Cache for user sessions and analysis state"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
    
    def create_session(self, session_id: str, session_data: Dict, ttl: int = 3600) -> bool:
        """Create user session"""
        if not self.cache.is_available():
            return False
        
        try:
            key = f"session:{session_id}"
            session_data['created_at'] = datetime.utcnow().isoformat()
            self.cache.redis.setex(key, ttl, json.dumps(session_data))
            return True
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        if not self.cache.is_available():
            return None
        
        try:
            key = f"session:{session_id}"
            data = self.cache.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return None
    
    def update_session(self, session_id: str, updates: Dict, ttl: int = 3600) -> bool:
        """Update session data"""
        if not self.cache.is_available():
            return False
        
        try:
            current = self.get_session(session_id)
            if current:
                current.update(updates)
                current['updated_at'] = datetime.utcnow().isoformat()
                key = f"session:{session_id}"
                self.cache.redis.setex(key, ttl, json.dumps(current))
                return True
            return False
        except Exception as e:
            logger.error(f"Session update failed: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if not self.cache.is_available():
            return False
        
        try:
            key = f"session:{session_id}"
            return self.cache.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Session deletion failed: {e}")
            return False
