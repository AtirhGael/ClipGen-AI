"""
Database connection utilities for Redis and PostgreSQL
"""
import redis
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Redis connection
redis_client: Optional[redis.Redis] = None

# PostgreSQL connection pool
postgres_pool: Optional[SimpleConnectionPool] = None

# SQLAlchemy engine and session
engine = None
SessionLocal = None


def init_redis() -> redis.Redis:
    """Initialize Redis connection"""
    global redis_client
    try:
        redis_client = redis.from_url(
            settings.redis_connection_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        # Test the connection
        redis_client.ping()
        logger.info("Redis connection established successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise


def init_postgres():
    """Initialize PostgreSQL connection pool and SQLAlchemy engine"""
    global postgres_pool, engine, SessionLocal
    
    try:
        # Create connection pool for direct psycopg2 connections
        postgres_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        
        # Create SQLAlchemy engine
        engine = create_engine(
            settings.postgres_connection_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("PostgreSQL connection established successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    if redis_client is None:
        logger.warning("Redis client not initialized - attempting to reconnect")
        try:
            return init_redis()
        except Exception as e:
            raise RuntimeError(f"Redis client not available: {e}")
    return redis_client


@contextmanager
def get_postgres_connection():
    """Get PostgreSQL connection from pool"""
    if postgres_pool is None:
        logger.warning("PostgreSQL pool not initialized - attempting to reconnect")
        try:
            init_postgres()
        except Exception as e:
            raise RuntimeError(f"PostgreSQL pool not available: {e}")
    
    conn = None
    try:
        conn = postgres_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            postgres_pool.putconn(conn)


def get_db() -> Generator:
    """Get SQLAlchemy database session (FastAPI dependency)"""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_postgres() first.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def check_redis_health() -> dict:
    """Check Redis connection health"""
    try:
        redis_conn = get_redis()
        
        # Test basic operations
        test_key = "health_check"
        redis_conn.set(test_key, "ok", ex=5)  # 5 second expiration
        result = redis_conn.get(test_key)
        redis_conn.delete(test_key)
        
        info = redis_conn.info()
        
        return {
            "status": "healthy",
            "connected": True,
            "test_result": result == "ok",
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown")
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


async def check_postgres_health() -> dict:
    """Check PostgreSQL connection health"""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                # Test basic query
                cursor.execute("SELECT version(), current_database(), current_user")
                version, database, user = cursor.fetchone()
                
                # Get connection info
                cursor.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
                active_connections = cursor.fetchone()[0]
        
        return {
            "status": "healthy",
            "connected": True,
            "postgres_version": version.split()[0] if version else "unknown",
            "database": database,
            "user": user,
            "active_connections": active_connections
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


def close_connections():
    """Close all database connections"""
    global redis_client, postgres_pool, engine
    
    try:
        if redis_client:
            redis_client.close()
            redis_client = None
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")
    
    try:
        if postgres_pool:
            postgres_pool.closeall()
            postgres_pool = None
            logger.info("PostgreSQL pool closed")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL pool: {e}")
    
    try:
        if engine:
            engine.dispose()
            engine = None
            logger.info("SQLAlchemy engine disposed")
    except Exception as e:
        logger.error(f"Error disposing SQLAlchemy engine: {e}")