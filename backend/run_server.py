"""
Simple FastAPI server runner for ClipGen-AI
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting ClipGen-AI FastAPI Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n✅ Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )