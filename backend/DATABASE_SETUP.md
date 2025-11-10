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

### Option 2: Using Chocolatey (Windows)
```bash
# Install Chocolatey from https://chocolatey.org/install
# Then run in PowerShell as Administrator:
choco install redis-64

# Start Redis:
redis-server
```

### Option 3: Using WSL2 (Windows Subsystem for Linux)
```bash
# Install WSL2 and Ubuntu
# Then in WSL terminal:
sudo apt update
sudo apt install redis-server

# Start Redis:
sudo service redis-server start
```

### Option 4: Manual Installation
1. Download Redis for Windows from: https://github.com/microsoftarchive/redis/releases
2. Extract and run `redis-server.exe`

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

### Option 3: Using Chocolatey
```bash
choco install postgresql
```

## Verification Commands

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
- ❌ Redis Server - Not running
- ❌ PostgreSQL Server - Not running

The FastAPI application will start successfully even without the database servers running, but database-related features will be unavailable until you set them up.