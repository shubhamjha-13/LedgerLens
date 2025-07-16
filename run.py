#!/usr/bin/env python3
"""
Startup script for Financial RAG System
Handles both backend API and frontend serving
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import chromadb
        import openai
        print(" All dependencies found")
        return True
    except ImportError as e:
        print(f" Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    print(" OpenAI API key found")
    return True

def check_required_files():
    """Check if required files exist"""
    required_files = [
        "financial_rag.py",
        "pdf_processor.py", 
        "app.py",
        "index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f" Missing required files: {', '.join(missing_files)}")
        return False
    
    print(" All required files found")
    return True

def start_backend(port=8000):
    """Start the FastAPI backend"""
    print(f" Starting backend on port {port}...")
    
    try:
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", str(port),
            "--reload"
        ])
        return process
    except Exception as e:
        print(f" Failed to start backend: {e}")
        return None

def start_frontend_server(port=3000):
    """Start a simple HTTP server for the frontend"""
    print(f" Starting frontend server on port {port}...")
    
    try:
        # Start simple HTTP server
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", str(port)
        ])
        return process
    except Exception as e:
        print(f" Failed to start frontend server: {e}")
        return None

def wait_for_server(url, timeout=30):
    """Wait for server to be ready"""
    import requests
    
    print(f" Waiting for server at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f" Server ready at {url}")
                return True
        except:
            pass
        time.sleep(1)
    
    print(f" Server not ready after {timeout} seconds")
    return False

def main():
    parser = argparse.ArgumentParser(description="Financial RAG System Startup")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend API port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies")
    
    args = parser.parse_args()
    
    print(" Financial RAG System Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if not check_api_key():
        return 1
    
    if not check_required_files():
        return 1
    
    if args.check_only:
        print(" All checks passed! System ready to start.")
        return 0
    
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend(args.backend_port)
        if not backend_process:
            return 1
        processes.append(backend_process)
        
        # Wait for backend to be ready
        backend_url = f"http://localhost:{args.backend_port}"
        if not wait_for_server(backend_url):
            return 1
        
        if not args.backend_only:
            # Start frontend server
            frontend_process = start_frontend_server(args.frontend_port)
            if frontend_process:
                processes.append(frontend_process)
                
                # Wait a bit for frontend server
                time.sleep(2)
                
                frontend_url = f"http://localhost:{args.frontend_port}"
                
                print("\n System started successfully!")
                print("=" * 50)
                print(f" Backend API: {backend_url}")
                print(f" Frontend: {frontend_url}")
                print(f" API Docs: {backend_url}/docs")
                print("\nPress Ctrl+C to stop the servers")
                
                # Open browser
                if not args.no_browser:
                    print(" Opening browser...")
                    webbrowser.open(frontend_url)
            else:
                print(f"\n  Frontend server failed to start, but backend is running at {backend_url}")
                print("You can:")
                print(f"1. Open {backend_url}/docs for API documentation")
                print("2. Serve index.html manually")
                print("3. Use the backend API directly")
        else:
            print(f"\n Backend started successfully at {backend_url}")
            print(f" API Docs: {backend_url}/docs")
            print("\nPress Ctrl+C to stop the server")
        
        # Wait for keyboard interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n Shutting down servers...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print(" Goodbye!")
        return 0
    
    except Exception as e:
        print(f" Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())