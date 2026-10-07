"""
===============================================================================
GeoFusion AI — Authoritative Root Entry Point
Delegates execution to the primary Streamlit application in app/app.py.
===============================================================================
"""

import importlib.util
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Dynamically load the authoritative application from app/app.py
_APP_MODULE_PATH = ROOT_DIR / "app" / "app.py"
_spec = importlib.util.spec_from_file_location("geofusion_main_app", _APP_MODULE_PATH)
_app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app_module)

# Expose all exported functions and attributes at module level for compatibility
globals().update({k: v for k, v in _app_module.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    _app_module.main()
