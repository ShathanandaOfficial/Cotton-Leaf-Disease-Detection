"""
API Routes package for Cotton Disease Detection
"""
from .predictions import router as predictions_router

__all__ = ["predictions_router"]