"""
AegisPCAP Dashboard FastAPI Application
Production-grade dashboard backend with real-time analytics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os
from pathlib import Path

from .endpoints import router as dashboard_router
from src.db.persistence import get_persistence_layer, initialize_persistence
from src.integrations.endpoints import router as integrations_router, initialize_integrations

# ============================================================================
# Configuration
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
API_TITLE = "AegisPCAP Dashboard API"
API_VERSION = "0.3.0"
API_DESCRIPTION = """
Enterprise-grade network security threat detection dashboard.

## Features
- Real-time threat monitoring
- Interactive network topology visualization
- Advanced threat analytics
- Integration with detection pipeline
"""

# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup and shutdown
    """
    # Startup
    logger.info("Initializing AegisPCAP Dashboard...")
    try:
        persistence = get_persistence_layer()
        logger.info("Persistence layer initialized successfully")
        
        # Initialize integrations
        initialize_integrations()
        logger.info("Integrations initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AegisPCAP Dashboard...")


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Local React dev
        "http://localhost:8080",      # Alternative frontend
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count"]
)

# Trust proxy headers (for production deployments)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", os.getenv("ALLOWED_HOSTS", "")]
)

# ============================================================================
# Custom Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AegisPCAP Dashboard API",
        "version": API_VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "redoc": "/redoc"
    }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": {
            "overview": "/api/dashboard/overview",
            "flows": "/api/dashboard/flows",
            "alerts": "/api/dashboard/alerts",
            "incidents": "/api/dashboard/incidents",
            "analytics": "/api/dashboard/analytics/*"
        },
        "documentation": "/docs"
    }


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(dashboard_router)
app.include_router(integrations_router)

# ============================================================================
# Static Files (Optional - for serving frontend)
# ============================================================================

# Uncomment when frontend is ready:
# static_dir = Path(__file__).parent / "static"
# if static_dir.exists():
#     app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    workers = int(os.getenv("DASHBOARD_WORKERS", "4"))
    
    logger.info(f"Starting AegisPCAP Dashboard on {host}:{port}")
    logger.info(f"Documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("ENV", "production") == "development",
        log_level="info"
    )
