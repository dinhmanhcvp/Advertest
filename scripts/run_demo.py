"""Runner script for AdverTest Web Demo."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)
    
    print("=" * 60)
    print("🚀 Starting AdverTest Interactive Web Demo...")
    print("=" * 60)
    print("The server will start shortly.")
    print("Open your browser and navigate to: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\nDemo server stopped.")

if __name__ == "__main__":
    main()
