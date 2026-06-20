"""scripts/start_api.py – Step 4: Start FastAPI server on port 8000"""

import sys, os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "api"))
sys.path.insert(0, _root) # Allow importing api.main
os.chdir(_root)   # ensure relative imports resolve

import uvicorn


if __name__ == "__main__":
    print("=" * 60)
    print("Starting FastAPI - Bank of Baroda ATO Detection API")
    print("Docs:  http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        app_dir=_root,
    )
