#!/usr/bin/env python
"""
THETA Server Runner
Run with: python run.py [--port PORT] [--host HOST] [--reload]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("/root/autodl-tmp/ETM")))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(description="Run THETA Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    import uvicorn
    
    print(f"Starting THETA server on {args.host}:{args.port}")
    print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
