"""
Configuration settings for ClipGen-AI Backend
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application settings
    app_name: str = "ClipGen-AI"
    version: str = "1.0.0"
    debug: bool = False
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_url: Optional[str] = None
    
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "clipgen"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_url: Optional[str] = None
    
    # Database URLs (constructed if not provided)
    @property
    def redis_connection_url(self) -> str:
        if self.redis_url:
            return self.redis_url
        
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def postgres_connection_url(self) -> str:
        if self.postgres_url:
            return self.postgres_url
        
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # File paths
    upload_dir: str = "uploads"
    download_dir: str = "downloads"
    temp_dir: str = "temp"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()