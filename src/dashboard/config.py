"""
Dashboard Configuration
Environment and deployment settings for the dashboard API
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    
    # Server Configuration
    host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    workers: int = int(os.getenv("DASHBOARD_WORKERS", "4"))
    environment: str = os.getenv("ENV", "production")
    
    # CORS Configuration
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ])
    cors_credentials: bool = True
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    
    # API Configuration
    api_title: str = "AegisPCAP Dashboard API"
    api_version: str = "0.3.0"
    api_prefix: str = "/api/dashboard"
    
    # Database Configuration
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "aegispcap")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    # Cache Configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Pagination Configuration
    default_page_size: int = 100
    max_page_size: int = 1000
    
    # Cache TTLs (seconds)
    flow_cache_ttl: int = 3600  # 1 hour
    alert_cache_ttl: int = 7200  # 2 hours
    stats_cache_ttl: int = 300  # 5 minutes
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = 30  # seconds
    ws_max_connections: int = 1000
    ws_timeout: int = 60  # seconds
    
    # Performance Configuration
    query_timeout: int = 30000  # milliseconds
    batch_size: int = 1000
    
    # Feature Flags
    enable_analytics: bool = True
    enable_websocket: bool = True
    enable_experimental_endpoints: bool = os.getenv("EXPERIMENTAL", "false").lower() == "true"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        
        if self.workers < 1 or self.workers > 32:
            raise ValueError(f"Invalid workers: {self.workers}")
        
        if self.default_page_size > self.max_page_size:
            raise ValueError("default_page_size cannot exceed max_page_size")


@dataclass
class DevelopmentConfig(DashboardConfig):
    """Development configuration"""
    environment: str = "development"
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://localhost:5000",
    ])
    log_level: str = "DEBUG"
    workers: int = 1


@dataclass
class ProductionConfig(DashboardConfig):
    """Production configuration"""
    environment: str = "production"
    cors_origins: list = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "").split(","))
    log_level: str = "WARNING"
    enable_experimental_endpoints: bool = False


@dataclass
class TestingConfig(DashboardConfig):
    """Testing configuration"""
    environment: str = "testing"
    port: int = 8888
    db_name: str = "aegispcap_test"
    log_level: str = "ERROR"
    workers: int = 1


# ============================================================================
# Global Configuration Instance
# ============================================================================

def get_config() -> DashboardConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENV", "production").lower()
    
    if env == "development":
        return DevelopmentConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return ProductionConfig()


# Initialize global config
config = get_config()
