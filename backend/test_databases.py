"""
Test script to verify Redis and PostgreSQL connections
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings
from app.database import init_redis, init_postgres, check_redis_health, check_postgres_health

async def test_databases():
    """Test both Redis and PostgreSQL connections"""
    print("🧪 Testing ClipGen-AI Database Connections")
    print("=" * 50)
    
    # Test Redis
    print("\n📦 Testing Redis Connection...")
    try:
        redis_client = init_redis()
        print("✅ Redis connection established successfully!")
        
        # Run health check
        redis_health = await check_redis_health()
        print(f"   Status: {redis_health['status']}")
        print(f"   Version: {redis_health.get('redis_version', 'Unknown')}")
        print(f"   Connected Clients: {redis_health.get('connected_clients', 0)}")
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
    
    # Test PostgreSQL
    print("\n🐘 Testing PostgreSQL Connection...")
    try:
        init_postgres()
        print("✅ PostgreSQL connection established successfully!")
        
        # Run health check
        postgres_health = await check_postgres_health()
        print(f"   Status: {postgres_health['status']}")
        print(f"   Database: {postgres_health.get('database', 'Unknown')}")
        print(f"   User: {postgres_health.get('user', 'Unknown')}")
        print(f"   Version: {postgres_health.get('postgres_version', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
    
    # Test configuration
    print(f"\n⚙️  Configuration Summary:")
    print(f"   Redis URL: {settings.redis_connection_url}")
    print(f"   PostgreSQL URL: {settings.postgres_connection_url}")
    print(f"   Upload Directory: {settings.upload_dir}")
    print(f"   Debug Mode: {settings.debug}")
    
    print("\n🎉 Database testing completed!")

if __name__ == "__main__":
    asyncio.run(test_databases())