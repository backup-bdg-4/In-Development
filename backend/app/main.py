import os
import json
import logging
import traceback
import time
from typing import List, Dict, Any, Optional
import numpy as np
import coremltools as ct
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import requests
import sys
import gc
import threading
from prometheus_fastapi_instrumentator import Instrumentator

# Import configuration and utilities
from .config import settings

# Model status tracking
model_status = {
    "loaded": True,
    "path": os.environ.get('MODEL_DATA_PATH', '/tmp/model'),
    "exists": True,
    "last_error": None,
    "load_attempts": 0,
    "last_attempt_time": None,
    "details": {},
    "alternate_paths": [],
    "search_paths_checked": []
}

# Global variables for Jupyter server
JUPYTER_SERVER_CHECK_THREAD = None
