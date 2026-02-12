"""
Prometheus metrics instrumentation for AegisPCAP FastAPI application.

This module provides:
- Application-level metrics (requests, responses, latency)
- Business metrics (flows processed, alerts generated, model inference)
- Infrastructure metrics (database queries, cache performance)
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_client.exposition import generate_latest
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse
from functools import wraps
import time
from typing import Callable, Any

# Create registry for metrics
registry = CollectorRegistry()

# ===== REQUEST METRICS =====

request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_latency = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

request_size = Histogram(
    'api_request_size_bytes',
    'API request body size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000),
    registry=registry
)

response_size = Histogram(
    'api_response_size_bytes',
    'API response body size in bytes',
    ['method', 'endpoint', 'status'],
    buckets=(100, 1000, 10000, 100000, 1000000),
    registry=registry
)

# ===== FLOW PROCESSING METRICS =====

flows_processed = Counter(
    'flows_processed_total',
    'Total flows processed',
    ['source', 'status'],
    registry=registry
)

flow_processing_time = Histogram(
    'flow_processing_seconds',
    'Time to process single flow in seconds',
    ['source'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=registry
)

flows_in_queue = Gauge(
    'flows_queue_depth',
    'Current number of flows in processing queue',
    registry=registry
)

# ===== MODEL METRICS =====

model_predictions = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model', 'prediction_class'],
    registry=registry
)

model_inference_time = Histogram(
    'model_inference_seconds',
    'Model inference latency in seconds',
    ['model'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy_score',
    'Model accuracy on test set',
    ['model'],
    registry=registry
)

model_confidence = Histogram(
    'model_prediction_confidence',
    'Model prediction confidence distribution',
    ['model'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# ===== ALERT METRICS =====

alerts_generated = Counter(
    'alerts_generated_total',
    'Total alerts generated',
    ['alert_type', 'severity'],
    registry=registry
)

alerts_deduplicated = Counter(
    'alerts_deduplicated_total',
    'Total alerts deduplicated',
    ['alert_type'],
    registry=registry
)

active_alerts = Gauge(
    'active_alerts_current',
    'Current number of active alerts',
    ['severity'],
    registry=registry
)

alert_processing_time = Histogram(
    'alert_processing_seconds',
    'Time to process alert',
    ['alert_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

# ===== DATABASE METRICS =====

db_query_time = Histogram(
    'db_query_seconds',
    'Database query duration in seconds',
    ['query_type', 'table'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=registry
)

db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['query_type', 'status'],
    registry=registry
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size',
    registry=registry
)

db_connection_pool_available = Gauge(
    'db_connection_pool_available',
    'Available connections in pool',
    registry=registry
)

# ===== CACHE METRICS =====

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_name'],
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_name'],
    registry=registry
)

cache_size = Gauge(
    'cache_size_bytes',
    'Cache size in bytes',
    ['cache_name'],
    registry=registry
)

cache_evictions = Counter(
    'cache_evictions_total',
    'Total cache evictions',
    ['cache_name'],
    registry=registry
)

# ===== FEATURE STORE METRICS =====

feature_store_queries = Counter(
    'feature_store_queries_total',
    'Total feature store queries',
    ['feature_version', 'status'],
    registry=registry
)

feature_store_latency = Histogram(
    'feature_store_latency_seconds',
    'Feature store query latency',
    ['feature_version'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

# ===== INTEGRATION METRICS =====

integration_calls = Counter(
    'integration_calls_total',
    'Total calls to external integrations',
    ['integration_type', 'status'],
    registry=registry
)

integration_latency = Histogram(
    'integration_latency_seconds',
    'External integration call latency',
    ['integration_type'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
    registry=registry
)

# ===== SYSTEM METRICS =====

system_uptime = Gauge(
    'system_uptime_seconds',
    'System uptime in seconds',
    registry=registry
)

data_pipeline_lag = Gauge(
    'data_pipeline_lag_seconds',
    'Data pipeline processing lag',
    registry=registry
)

# ===== MIDDLEWARE =====

async def prometheus_middleware(request: Request, call_next: Callable) -> FastAPIResponse:
    """Middleware to collect request metrics."""
    method = request.method
    endpoint = request.url.path
    
    # Record request size
    if request.headers.get('content-length'):
        request_size.labels(method=method, endpoint=endpoint).observe(
            float(request.headers['content-length'])
        )
    
    # Track request timing
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as exc:
        # Record failed request
        request_count.labels(method=method, endpoint=endpoint, status='error').inc()
        raise
    
    # Calculate latency
    latency = time.time() - start_time
    request_latency.labels(method=method, endpoint=endpoint).observe(latency)
    
    # Record response metrics
    request_count.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()
    
    # Record response size (estimate from content-length header)
    if response.headers.get('content-length'):
        response_size.labels(
            method=method,
            endpoint=endpoint,
            status=response.status_code
        ).observe(float(response.headers['content-length']))
    
    return response


# ===== DECORATORS FOR INSTRUMENTATION =====

def instrument_flow_processing(func: Callable) -> Callable:
    """Decorator to instrument flow processing functions."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            status = 'success'
            return result
        except Exception as exc:
            status = 'error'
            raise
        finally:
            latency = time.time() - start_time
            source = kwargs.get('source', 'unknown')
            flows_processed.labels(source=source, status=status).inc()
            flow_processing_time.labels(source=source).observe(latency)
    return wrapper


def instrument_model_inference(model_name: str):
    """Decorator for model inference instrumentation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record inference time
                latency = time.time() - start_time
                model_inference_time.labels(model=model_name).observe(latency)
                
                # Record prediction class if available
                if isinstance(result, dict) and 'prediction' in result:
                    prediction_class = result.get('prediction', 'unknown')
                    model_predictions.labels(
                        model=model_name,
                        prediction_class=str(prediction_class)
                    ).inc()
                    
                    # Record confidence if available
                    if 'confidence' in result:
                        model_confidence.labels(model=model_name).observe(
                            float(result['confidence'])
                        )
                
                return result
            except Exception as exc:
                raise
        return wrapper
    return decorator


def instrument_db_query(query_type: str, table: str = 'unknown'):
    """Decorator for database query instrumentation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                db_queries_total.labels(query_type=query_type, status='success').inc()
                return result
            except Exception as exc:
                db_queries_total.labels(query_type=query_type, status='error').inc()
                raise
            finally:
                latency = time.time() - start_time
                db_query_time.labels(query_type=query_type, table=table).observe(latency)
        return wrapper
    return decorator


# ===== METRICS ENDPOINT =====

def get_metrics() -> tuple[str, str]:
    """Generate Prometheus metrics in text format."""
    return (generate_latest(registry).decode('utf-8'), CONTENT_TYPE_LATEST)
