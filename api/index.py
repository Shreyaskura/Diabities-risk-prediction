"""
Vercel Serverless Function entry point for FastAPI backend.
Maps incoming requests to the DiaRisk ML FastAPI application.
"""

import sys
import os

# Add root directory and backend directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(root_dir, 'backend')

for path in [root_dir, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.main import app

# Vercel entrypoint
handler = app
