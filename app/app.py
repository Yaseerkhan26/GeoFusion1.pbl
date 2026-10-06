"""
===============================================================================
File: app/app.py
Description: Authoritative pass-through entry point to the primary FusionLand AI
             satellite intelligence application (located at root /app.py).
===============================================================================
"""

import sys
from pathlib import Path

# Append project root to system path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import authoritative main application
import app

if __name__ == "__main__":
    app.main()
