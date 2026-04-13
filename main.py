"""
TB DETECTOR - Vercel Entrypoint
FastAPI application entry point for Vercel deployment
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import and expose the FastAPI app
from main_v3 import app

# Export for Vercel
__all__ = ["app"]
