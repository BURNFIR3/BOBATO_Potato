"""scripts/run_tests.py – Run all pytest tests"""

import sys, os, subprocess

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tests = os.path.join(_root, "tests")

if __name__ == "__main__":
    print("=" * 60)
    print("Running ATO Detection Test Suite")
    print("=" * 60)
    result = subprocess.run([
        sys.executable, "-m", "pytest", tests, "-v",
        "--tb=short", "--no-header",
    ])
    sys.exit(result.returncode)
