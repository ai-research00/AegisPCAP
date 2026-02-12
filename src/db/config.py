"""
AegisPCAP Database configuration - connection strings and setup
"""
import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Database engine types
class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


@dataclass
class PostgreSQLConfig:
    """PostgreSQL database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', 5432))
    database: str = os.getenv('DB_NAME', 'aegis_pcap')
    user: str = os.getenv('DB_USER', 'aegis_user')
    password: str = os.getenv('DB_PASSWORD', 'secure_password')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', 20))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', 40))
    echo: bool = os.getenv('DB_ECHO', 'False').lower() == 'true'
    
    def get_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class MySQLConfig:
    """MySQL database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', 3306))
    database: str = os.getenv('DB_NAME', 'aegis_pcap')
    user: str = os.getenv('DB_USER', 'aegis_user')
    password: str = os.getenv('DB_PASSWORD', 'secure_password')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', 20))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', 40))
    echo: bool = os.getenv('DB_ECHO', 'False').lower() == 'true'
    
    def get_url(self) -> str:
        """Generate MySQL connection URL"""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class SQLiteConfig:
    """SQLite database configuration"""
    path: str = os.getenv('DB_PATH', './aegis_pcap.db')
    echo: bool = os.getenv('DB_ECHO', 'False').lower() == 'true'
    
    def get_url(self) -> str:
        """Generate SQLite connection URL"""
        return f"sqlite:///{self.path}"


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', 6379))
    db: int = int(os.getenv('REDIS_DB', 0))
    password: Optional[str] = os.getenv('REDIS_PASSWORD', None)
    ssl: bool = os.getenv('REDIS_SSL', 'False').lower() == 'true'
    max_connections: int = int(os.getenv('REDIS_MAX_CONN', 50))
    socket_timeout: int = int(os.getenv('REDIS_TIMEOUT', 5))
    
    def get_url(self) -> str:
        """Generate Redis connection URL"""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class MongoDBConfig:
    """MongoDB configuration"""
    host: str = os.getenv('MONGODB_HOST', 'localhost')
    port: int = int(os.getenv('MONGODB_PORT', 27017))
    database: str = os.getenv('MONGODB_NAME', 'aegis_pcap')
    user: Optional[str] = os.getenv('MONGODB_USER', None)
    password: Optional[str] = os.getenv('MONGODB_PASSWORD', None)
    replica_set: Optional[str] = os.getenv('MONGODB_REPLICA_SET', None)
    
    def get_url(self) -> str:
        """Generate MongoDB connection URL"""
        if self.user and self.password:
            auth = f"{self.user}:{self.password}@"
        else:
            auth = ""
        
        url = f"mongodb://{auth}{self.host}:{self.port}/{self.database}"
        
        if self.replica_set:
            url += f"?replicaSet={self.replica_set}"
        
        return url


@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration"""
    host: str = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    port: int = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    user: Optional[str] = os.getenv('ELASTICSEARCH_USER', None)
    password: Optional[str] = os.getenv('ELASTICSEARCH_PASSWORD', None)
    scheme: str = os.getenv('ELASTICSEARCH_SCHEME', 'http')
    index_prefix: str = 'aegis'
    
    def get_url(self) -> str:
        """Generate Elasticsearch URL"""
        if self.user and self.password:
            auth = f"{self.user}:{self.password}@"
        else:
            auth = ""
        return f"{self.scheme}://{auth}{self.host}:{self.port}"


class DatabaseConfig:
    """Central database configuration manager"""
    
    # Database type
    db_type: DatabaseType = DatabaseType[os.getenv('DATABASE_TYPE', 'POSTGRESQL')]
    
    # Primary database
    postgresql = PostgreSQLConfig()
    mysql = MySQLConfig()
    sqlite = SQLiteConfig()
    
    # Secondary storage
    mongodb = MongoDBConfig()
    
    # Caching
    redis = RedisConfig()
    
    # Search
    elasticsearch = ElasticsearchConfig()
    
    @classmethod
    def get_primary_db_url(cls) -> str:
        """Get primary database URL"""
        if cls.db_type == DatabaseType.POSTGRESQL:
            return cls.postgresql.get_url()
        elif cls.db_type == DatabaseType.MYSQL:
            return cls.mysql.get_url()
        elif cls.db_type == DatabaseType.SQLITE:
            return cls.sqlite.get_url()
        else:
            return cls.sqlite.get_url()  # Default to SQLite
    
    @classmethod
    def get_cache_url(cls) -> str:
        """Get cache (Redis) URL"""
        return cls.redis.get_url()
    
    @classmethod
    def get_search_url(cls) -> str:
        """Get search (Elasticsearch) URL"""
        return cls.elasticsearch.get_url()
    
    @classmethod
    def get_document_db_url(cls) -> str:
        """Get document database (MongoDB) URL"""
        return cls.mongodb.get_url()


# Connection pool configurations for different scenarios
POOL_CONFIGS = {
    'default': {
        'pool_size': 20,
        'max_overflow': 40,
        'pool_recycle': 3600,
    },
    'high_performance': {
        'pool_size': 50,
        'max_overflow': 100,
        'pool_recycle': 1800,
    },
    'low_resource': {
        'pool_size': 5,
        'max_overflow': 10,
        'pool_recycle': 7200,
    },
    'testing': {
        'pool_size': 1,
        'max_overflow': 0,
    },
}


# Feature store configuration for different storage backends
FEATURE_STORE_BACKENDS = {
    'redis': {
        'type': 'redis',
        'ttl': 3600,  # 1 hour
        'compression': True,
    },
    'postgresql': {
        'type': 'postgresql',
        'table': 'feature_store',
        'partition_by': 'time_window',
    },
    'mongodb': {
        'type': 'mongodb',
        'collection': 'feature_store',
        'ttl': 86400,  # 1 day
    },
    'elasticsearch': {
        'type': 'elasticsearch',
        'index': 'aegis-features-*',
        'shard_count': 3,
        'replica_count': 1,
    },
}


# Data retention policies (in days)
DATA_RETENTION_POLICIES = {
    'flows': 90,              # Raw flows: 90 days
    'features': 180,          # Extracted features: 180 days
    'alerts': 365,            # Alerts: 1 year
    'incidents': 730,         # Incidents: 2 years
    'audit_logs': 365,        # Audit logs: 1 year
    'threat_intelligence': 180,  # Threat intel: 180 days
    'models': -1,             # Models: indefinite
}


# Backup configuration
BACKUP_CONFIG = {
    'enabled': os.getenv('DB_BACKUP_ENABLED', 'True').lower() == 'true',
    'schedule': os.getenv('DB_BACKUP_SCHEDULE', '0 2 * * *'),  # Daily at 2 AM
    'retention_days': int(os.getenv('DB_BACKUP_RETENTION', 30)),
    'location': os.getenv('DB_BACKUP_LOCATION', '/var/backups/aegis-pcap/'),
    'compress': os.getenv('DB_BACKUP_COMPRESS', 'True').lower() == 'true',
}


# Query optimization settings
QUERY_OPTIMIZATION = {
    'batch_insert_size': 1000,
    'batch_update_size': 500,
    'pagination_limit': 100,
    'max_query_timeout': 30,  # seconds
    'enable_query_cache': True,
    'cache_ttl': 300,  # 5 minutes
}


# High availability configuration
HA_CONFIG = {
    'enabled': os.getenv('HA_ENABLED', 'False').lower() == 'true',
    'replica_count': int(os.getenv('HA_REPLICAS', 2)),
    'failover_timeout': int(os.getenv('HA_FAILOVER_TIMEOUT', 30)),
    'health_check_interval': int(os.getenv('HA_HEALTH_CHECK', 10)),
}
