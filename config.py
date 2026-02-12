"""
AegisPCAP Configuration - Central settings for all components
"""

import os
from pathlib import Path
from typing import Dict, List

# ============================================================================
# Project Configuration
# ============================================================================

PROJECT_NAME = "AegisPCAP"
VERSION = "0.3.0"
DESCRIPTION = "AI-Agent-Driven Network Traffic Intelligence Platform"
AUTHOR = "Security Team"

# ============================================================================
# Paths Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
PCAP_DIR = DATA_DIR / "raw_pcaps"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PCAP_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database Configuration
# ============================================================================

# PostgreSQL (Primary database)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "aegispcap")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 20))

SQLALCHEMY_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# MongoDB (Feature store / Document storage)
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB = os.getenv("MONGO_DB", "aegispcap")
MONGO_URL = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"

# Redis (Caching)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Elasticsearch (Alerts & logging)
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX_PREFIX = "aegispcap"

# ============================================================================
# PCAP Processing Configuration
# ============================================================================

# Packet parsing
MAX_PACKET_SIZE = 65535
MIN_PACKET_SIZE = 20

# Flow aggregation
FLOW_TIMEOUT = 300  # seconds - timeout for inactive flows
MIN_PACKETS_PER_FLOW = 1

# Data quality
VALIDATE_PACKETS = True
QUALITY_CHECK_ENABLED = True
DROP_MALFORMED_PACKETS = True

# GeoIP
GEOIP_DB_PATH = os.getenv("GEOIP_DB_PATH", None)  # Path to MaxMind GeoLite2 DB

# ============================================================================
# Feature Engineering Configuration
# ============================================================================

FEATURE_GROUPS = {
    "statistical": True,
    "timing": True,
    "dns": True,
    "tls": True,
    "quic": True,
    "protocol": True,
    "behavioral": True,
    "graph": False  # Requires additional processing
}

# Feature scaling
SCALE_FEATURES = True
SCALER_TYPE = "standard"  # 'standard' or 'robust'

# ============================================================================
# Machine Learning Configuration
# ============================================================================

# Ensemble models
ML_ENSEMBLE_ENABLED = True
ANOMALY_CONTAMINATION = 0.05  # Expected contamination rate

# Isolation Forest
IF_N_ESTIMATORS = 100
IF_MAX_SAMPLES = 256
IF_RANDOM_STATE = 42

# One-Class SVM
OCSVM_NU = 0.05
OCSVM_GAMMA = "auto"

# Local Outlier Factor
LOF_N_NEIGHBORS = 20

# Random Forest (for supervised)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.1

# Deep Learning
DL_BATCH_SIZE = 32
DL_EPOCHS = 20
DL_LEARNING_RATE = 0.001
DL_DROPOUT = 0.2

# LSTM parameters
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 2

# ============================================================================
# Threat Detection Thresholds
# ============================================================================

RISK_SCORE_THRESHOLDS = {
    "critical": 0.9,
    "high": 0.7,
    "medium": 0.5,
    "low": 0.3,
    "benign": 0.0
}

CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.8,
    "medium_confidence": 0.6,
    "low_confidence": 0.4
}

# Threat-specific thresholds
C2_THRESHOLD = 0.65
EXFIL_THRESHOLD = 0.60
BOTNET_THRESHOLD = 0.70
DGA_THRESHOLD = 0.65
BEACONING_THRESHOLD = 0.70

# ============================================================================
# API Configuration
# ============================================================================

# REST API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))

# Authentication
AUTH_ENABLED = True
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# CORS
CORS_ORIGINS = ["*"]  # Restrict in production
CORS_CREDENTIALS = True

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "aegispcap.log"
LOG_MAX_BYTES = 10485760  # 10MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# Monitoring & Observability
# ============================================================================

# Metrics
PROMETHEUS_ENABLED = True
PROMETHEUS_PORT = 8001

# Tracing
TRACING_ENABLED = False
TRACING_BACKEND = "jaeger"  # 'jaeger' or 'zipkin'
TRACING_SAMPLE_RATE = 0.1

# ============================================================================
# Integration Configuration
# ============================================================================

# Threat Intelligence APIs
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
SHODAN_API_KEY = os.getenv("SHODAN_API_KEY", "")
OTXAPI_KEY = os.getenv("OTXAPI_KEY", "")

# SOAR Integration
SOAR_ENABLED = False
SOAR_PLATFORM = "splunk_soar"  # or 'demisto', 'tines'
SOAR_API_URL = os.getenv("SOAR_API_URL", "")
SOAR_API_KEY = os.getenv("SOAR_API_KEY", "")

# SIEM Integration
SIEM_ENABLED = False
SIEM_TYPE = "splunk"  # or 'elk', 'wazuh'
SIEM_HOST = os.getenv("SIEM_HOST", "")
SIEM_PORT = int(os.getenv("SIEM_PORT", 8089))

# Firewall Integration
FIREWALL_ENABLED = False
FIREWALL_TYPE = "palo_alto"  # or 'fortinet', 'checkpoint'
FIREWALL_API_KEY = os.getenv("FIREWALL_API_KEY", "")

# Notifications
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")
EMAIL_ENABLED = False
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")

# ============================================================================
# Incident Response Configuration
# ============================================================================

# Auto-response
AUTO_RESPONSE_ENABLED = False
AUTO_RESPONSE_MIN_RISK = 0.8

# Actions
ACTIONS = {
    "block_ip": False,
    "block_domain": False,
    "isolate_endpoint": False,
    "create_ticket": True,
    "send_alert": True,
    "kill_process": False  # Dangerous - requires high confidence
}

# ============================================================================
# Compliance Configuration
# ============================================================================

COMPLIANCE_ENABLED = True
GDPR_MODE = True  # Anonymize sensitive data
DATA_RETENTION_DAYS = 90
LOG_RETENTION_DAYS = 365

# PII Masking
ANONYMIZE_IPS = True
IP_MASK_LEVEL = 4  # Mask last octet for IPv4
ANONYMIZE_DOMAINS = False

# ============================================================================
# Performance Tuning
# ============================================================================

# Batch processing
BATCH_SIZE_PCAP = 1000  # packets per batch
BATCH_SIZE_FLOWS = 100
BATCH_SIZE_INFERENCE = 64

# Caching
CACHE_ENABLED = True
CACHE_TTL_SECONDS = 3600
CACHE_MAX_SIZE_MB = 500

# Parallelization
N_WORKERS = 4  # Number of worker processes
USE_GPU = False  # Enable GPU acceleration

# ============================================================================
# Advanced Configuration
# ============================================================================

# Feature Store
FEATURE_STORE_ENABLED = True
FEATURE_STORE_BACKEND = "postgres"  # or 'mongodb'

# Model Registry
MODEL_REGISTRY_ENABLED = True
MODEL_REGISTRY_BACKEND = "mlflow"  # or 'custom'

# Active Learning
ACTIVE_LEARNING_ENABLED = False
AL_UNCERTAINTY_THRESHOLD = 0.5
AL_BATCH_SIZE = 10

# Federated Learning
FEDERATED_LEARNING_ENABLED = False
FL_NUM_ROUNDS = 10

# ============================================================================
# Debug Configuration
# ============================================================================

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
TESTING = False
VERBOSE = DEBUG

# Feature/model debugging
DEBUG_FEATURES = False
DEBUG_MODELS = False
DEBUG_AGENT = False

# Sample data for testing
USE_SAMPLE_PCAP = True
SAMPLE_PCAP_PATH = PCAP_DIR / "sample.pcap"
