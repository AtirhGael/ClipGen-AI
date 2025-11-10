"""
Database initialization script
"""
import logging
from sqlalchemy import create_engine
from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)


def init_database():
    """Initialize database tables"""
    try:
        # Create engine
        engine = create_engine(
            settings.postgres_connection_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        return engine
        
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise


if __name__ == "__main__":
    init_database()