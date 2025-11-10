# Database Setup Guide for ClipGen-AI

## Redis Setup Options

### Option 1: Using Docker (Recommended)
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop/
# Then run:
docker run --name redis-clipgen -p 6379:6379 -d redis:latest

# To stop Redis:
docker stop redis-clipgen

# To start Redis again:
docker start redis-clipgen
```

```bash
# Install WSL2 and Ubuntu
# Then in WSL terminal:
sudo apt update
sudo apt install redis-server

# Start Redis:
sudo service redis-server start
```

## PostgreSQL Setup Options

### Option 1: Using Docker (Recommended)
```bash
docker run --name postgres-clipgen \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=clipgen \
  -p 5432:5432 \
  -d postgres:latest
```

### Option 2: PostgreSQL Official Installer
1. Download from: https://www.postgresql.org/download/windows/
2. Install with default settings
3. Use password: `password` (or update .env file)
4. Create database: `clipgen`


### Test Redis Connection
```bash
# In PowerShell/CMD:
redis-cli ping
# Should return: PONG
```

### Test PostgreSQL Connection
```bash
# In PowerShell/CMD (after installing PostgreSQL client):
psql -h localhost -U postgres -d clipgen
# Enter password when prompted
```

## Environment Configuration

Update the `.env` file in the `backend` directory with your actual database credentials:

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_password_if_needed

# PostgreSQL Configuration  
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=clipgen
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_actual_password
```

## Current Status

- ✅ Python Redis client (`redis==7.0.1`) - Installed
- ✅ PostgreSQL client (`psycopg2-binary==2.9.11`) - Installed  
- ✅ Redis Server - Running (Docker: redis-clipgen)
- ✅ PostgreSQL Server - Running (Docker: my_postgres)
- ✅ FastAPI Backend - Connected to both databases

## Your Docker Containers

### Redis Container
```bash
Container ID: 813737f4fecf
Image: redis:latest  
Port: 0.0.0.0:6379->6379/tcp
Name: redis-clipgen
Status: Up 19 minutes
```

### PostgreSQL Container  
```bash
Container ID: bef4e21b08d9
Image: postgres
Port: 0.0.0.0:5432->5432/tcp
Name: my_postgres  
Status: Up 7 seconds
Database: clipgen
User: postgres
Password: password
```

## Container Management

### Start containers (if stopped):
```bash
docker start redis-clipgen
docker start my_postgres
```

### Stop containers:
```bash
docker stop redis-clipgen  
docker stop my_postgres
```

### View container logs:
```bash
docker logs redis-clipgen
docker logs my_postgres
```

All database connections are now working perfectly! 🎉