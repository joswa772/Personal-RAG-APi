#!/usr/bin/env python3
"""
FastAPI Server Startup Script

This script checks dependencies and starts the FastAPI server
with helpful status messages and error handling.
"""

import sys
import subprocess
import importlib.util
import os
from pathlib import Path

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ Missing dependency: {package_name}")
        return False
    else:
        print(f"✅ {package_name} is available")
        return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models")
            return True
        else:
            print("⚠️  Ollama is not responding properly")
            return False
    except:
        print("❌ Ollama is not running (optional - needed for Llama 3 and image generation)")
        return False

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_fastapi.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def main():
    """Main startup function"""
    print("🚀 PDF Processing & Image Generation - FastAPI Server")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("ai.py").exists():
        print("❌ Error: ai.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    required_deps = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("fitz", "PyMuPDF"),
        ("nltk", "NLTK"),
        ("sklearn", "Scikit-learn"),
        ("requests", "Requests"),
        ("PIL", "Pillow"),
    ]
    
    missing_deps = []
    for module, package in required_deps:
        if not check_dependency(module, package):
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install them manually:")
                print("pip install -r requirements_fastapi.txt")
                sys.exit(1)
        else:
            print("❌ Please install dependencies before running the server")
            sys.exit(1)
    
    # Check Ollama (optional)
    print("\n🤖 Checking Ollama...")
    ollama_available = check_ollama()
    
    # Check API key
    print("\n🔑 Checking Stable Diffusion API key...")
    try:
        from ai import STABLE_DIFFUSION_API_KEY
        if STABLE_DIFFUSION_API_KEY and not STABLE_DIFFUSION_API_KEY.startswith("sk-j6vcnrD7PvJNOxJXcrrg7QxA7aHHgsVr13DiXtrOAuV81hxt"):
            print("✅ Stable Diffusion API key is configured")
        else:
            print("⚠️  Stable Diffusion API key not configured (optional)")
    except:
        print("⚠️  Could not check API key configuration")
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    print(f"✅ Uploads directory ready: {uploads_dir}")
    
    # Start the server
    print("\n🚀 Starting FastAPI server...")
    print("=" * 60)
    print("📱 Web UI will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔧 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        import uvicorn
        uvicorn.run(
            "fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_fastapi.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 