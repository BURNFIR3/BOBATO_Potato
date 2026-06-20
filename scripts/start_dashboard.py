"""scripts/start_dashboard.py – Step 6: Start Streamlit dashboard"""

import sys
import os
import subprocess

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dashboard = os.path.join(_root, "dashboard", "app.py")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Streamlit ATO Dashboard")
    print("URL: http://localhost:8501")
    print("=" * 60)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--theme.base", "light",
    ])
