import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Brain Tumor 3D Visualization Server")
    print("=" * 60)
    print()
    print("Starting server...")
    print("Open http://localhost:8000 in your browser")
    print()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

