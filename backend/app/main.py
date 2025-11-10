from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from sqlalchemy import create_engine

from app.config import settings
from app.database import (
    init_redis, init_postgres, close_connections,
    check_redis_health, check_postgres_health,
    get_db, engine
)
from app.models import Base
from app.routers import videos

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up ClipGen-AI Backend...")
    
    # Try to initialize database connections (gracefully handle failures)
    logger.info("Attempting to initialize database connections...")
    
    try:
        init_redis()
        logger.info("✅ Redis connection initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.info("Application will continue without Redis")
    
    try:
        init_postgres()
        logger.info("✅ PostgreSQL connection initialized successfully")
        
        # Create database tables
        if engine:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables created/verified successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL connection failed: {e}")
        logger.info("Application will continue without PostgreSQL")
    
    logger.info("🚀 ClipGen-AI Backend startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ClipGen-AI Backend...")
    try:
        close_connections()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    logger.info("👋 ClipGen-AI Backend shutdown completed")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Video Highlight Generator",
    version=settings.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(videos.router)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ClipGen-AI Backend",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive system health check endpoint"""
    
    # Check if data directories exist
    upload_dir = Path(settings.upload_dir)
    download_dir = Path(settings.download_dir)
    temp_dir = Path(settings.temp_dir)
    
    # Create directories if they don't exist
    upload_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check database health
    redis_health = await check_redis_health()
    postgres_health = await check_postgres_health()
    
    return {
        "status": "healthy",
        "service": settings.app_name,
        "timestamp": datetime.now().isoformat(),
        "version": settings.version,
        "environment": "development" if settings.debug else "production",
        "databases": {
            "redis": redis_health,
            "postgres": postgres_health
        },
        "directories": {
            "uploads": {
                "path": str(upload_dir.absolute()),
                "exists": upload_dir.exists(),
                "writable": os.access(upload_dir, os.W_OK) if upload_dir.exists() else False
            },
            "downloads": {
                "path": str(download_dir.absolute()),
                "exists": download_dir.exists(),
                "writable": os.access(download_dir, os.W_OK) if download_dir.exists() else False
            },
            "temp": {
                "path": str(temp_dir.absolute()),
                "exists": temp_dir.exists(),
                "writable": os.access(temp_dir, os.W_OK) if temp_dir.exists() else False
            }
        },
        "system_info": {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name
        }
    }


@app.get("/health/redis")
async def redis_health_check():
    """Redis-specific health check endpoint"""
    return await check_redis_health()


@app.get("/health/postgres") 
async def postgres_health_check():
    """PostgreSQL-specific health check endpoint"""
    return await check_postgres_health()


@app.get("/config")
def get_config():
    """Get current application configuration (non-sensitive data only)"""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "debug": settings.debug,
        "redis_host": settings.redis_host,
        "redis_port": settings.redis_port,
        "redis_db": settings.redis_db,
        "postgres_host": settings.postgres_host,
        "postgres_port": settings.postgres_port,
        "postgres_db": settings.postgres_db,
        "postgres_user": settings.postgres_user,
        "upload_dir": settings.upload_dir,
        "download_dir": settings.download_dir,
        "temp_dir": settings.temp_dir
    }
