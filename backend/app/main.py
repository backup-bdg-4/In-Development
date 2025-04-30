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
from prometheus_fastapi_instrumentator import Instrumentator

# Import configuration and utilities
from .config import settings
from .utils.security_utils import setup_security, get_client_info, rate_limit_ip_and_tokens
from .utils.code_utils import contains_code, is_code_question, format_code_for_response
from .utils.response_utils import enhance_response, categorize_query, log_query_response
from .utils.search_utils import search_web, SEARCH_CACHE

# Configure logging based on settings
logging_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format=settings.logging.format,
    filename=settings.logging.log_file if settings.logging.log_to_file else None
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title=settings.app_name,
    description="An AI-powered conversational API using Apple's CoreML for natural language processing and question answering",
    version=settings.version,
    debug=settings.debug,
    docs_url="/api/docs" if settings.debug or settings.environment != "production" else None,
    redoc_url="/api/redoc" if settings.debug or settings.environment != "production" else None
)

# Add CORS middleware with origins from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup security features (rate limiting, secure headers, etc.)
setup_security(app)

# Add metrics instrumentation if not in production
if settings.environment != "production":
    Instrumentator().instrument(app).expose(app)

# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} version {settings.version} in {settings.environment} mode")
    
    # Import nltk resources if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        logger.info("NLTK resources loaded")
    except Exception as e:
        logger.warning(f"Failed to load NLTK resources: {e}")
    
    # Initialize the application with enhanced setup
    try:
        from .utils.initialization import initialize_app
        init_result = initialize_app()
        
        logger.info(f"Application initialized in {init_result['initialization_time_sec']:.2f} seconds")
        logger.info(f"Environment: {init_result['environment']['environment']}")
        
        if init_result['model']['found']:
            logger.info(f"✅ Model found at {init_result['model']['path']} ({init_result['model']['size_mb']:.2f} MB)")
        else:
            logger.error(f"❌ Model not found during initialization")
            
        # Save initialization info for health check
        app.state.init_result = init_result
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}")
        logger.error(traceback.format_exc())
        app.state.init_result = {
            "success": False,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Check initial model status
    global model, model_loaded
    if model_loaded:
        logger.info("Model pre-loaded successfully")
    else:
        logger.warning("Model will be loaded on first request")
        
        # Try to pre-load the model asynchronously
        import threading
        
        def preload_model_thread():
            try:
                global model, model_loaded
                logger.info("Starting asynchronous model pre-loading")
                model_loaded = load_model()
                if model_loaded:
                    logger.info("Model pre-loaded successfully in background thread")
                else:
                    logger.error("Failed to pre-load model in background thread")
            except Exception as e:
                logger.error(f"Error in model pre-loading thread: {str(e)}")
        
        # Start pre-loading in a background thread to avoid blocking startup
        preload_thread = threading.Thread(target=preload_model_thread)
        preload_thread.daemon = True
        preload_thread.start()
        logger.info("Started background thread for model pre-loading")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")
    
    # Clear any global resources
    if 'model' in globals() and model is not None:
        try:
            model = None
            logger.info("Model released from memory")
        except Exception as e:
            logger.error(f"Error releasing model: {e}")

# Add redirect from root to docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    if settings.debug or settings.environment != "production":
        return RedirectResponse(url="/api/docs")
    else:
        return {"status": "healthy", "service": settings.app_name}

# Import model utilities
from .utils.model_utils import find_model_file, validate_model, ensure_model_availability

# Global variable to store the loaded model
model = None

# Find model path from best available location
model_found, MODEL_PATH = find_model_file()

# Model status tracking - enhanced with multi-path support
model_status = {
    "loaded": False,
    "path": MODEL_PATH,
    "exists": model_found,
    "last_error": None,
    "load_attempts": 0,
    "last_attempt_time": None,
    "details": {},
    "alternate_paths": [],
    "search_paths_checked": []
}

def load_model(force_reload=False):
    """
    Load the CoreML model for inference.
    Memory-efficient version that minimizes redundant copies and validations.
    
    Args:
        force_reload (bool): If True, reload the model even if it's already loaded
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model, model_status
    
    # Check if we're in memory-saving mode
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    running_on_render = os.environ.get('RUNNING_ON_RENDER') == 'true'
    memory_saving_mode = minimize_memory or running_on_render
    
    # Track attempt
    model_status["load_attempts"] += 1
    model_status["last_attempt_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # If model is already loaded and no force reload, return True
    if model is not None and not force_reload:
        logger.info("Model already loaded, skipping load")
        return True
    
    # First, ensure model is available using our enhanced model utilities
    # This will use memory-saving mode if enabled
    model_availability = ensure_model_availability()
    
    # Update status
    model_status["exists"] = model_availability["found"]
    model_status["path"] = model_availability.get("source_path", MODEL_PATH)
    model_status["memory_saving_mode"] = memory_saving_mode
    
    # Store all paths we checked
    if "copied_to" in model_availability:
        model_status["alternate_paths"] = model_availability["copied_to"]
    
    # If model availability check failed, show clear error message
    if not model_availability["success"]:
        error_msg = f"""
=================================================================
ERROR: CoreML model file not available
=================================================================
The model file could not be found in any of these locations:
- {MODEL_PATH} (primary location)
- {', '.join(model_status.get('search_paths_checked', []))}

This model file should be stored using Git LFS in the repository.
If you're not seeing the file, make sure:

1. You have Git LFS installed: https://git-lfs.github.com
2. You've pulled the repository with Git LFS enabled:
   git lfs pull

Error details: {model_availability.get('error', 'Unknown error')}
=================================================================
"""
        logger.error(error_msg)
        model_status["last_error"] = f"Model file not found. Please ensure the CoreML model is properly installed."
        
        # Try downloading the model as a last resort
        try:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            from download_model import download_model
            
            logger.info("Attempting to download model...")
            if download_model():
                logger.info("Model downloaded successfully, retrying model availability check")
                model_availability = ensure_model_availability()
                
                if model_availability["success"]:
                    model_status["path"] = model_availability["source_path"]
                    model_status["exists"] = True
                    model_status["alternate_paths"] = model_availability.get("copied_to", [])
                else:
                    logger.error("Model downloaded but still not available")
                    return False
            else:
                logger.error("Failed to download model")
                return False
        except ImportError:
            logger.error("Could not import download_model module")
            return False
        except Exception as e:
            logger.error(f"Error during model download: {str(e)}")
            return False
    
    # Now that we have ensured model availability, continue with loading
    model_path = model_status["path"]
    
    # Load the ML model
    try:
        # Clear any previous model from memory
        if model is not None:
            model = None
        
        logger.info(f"Loading model from {model_path}")
        
        # Try importing coremltools if not already imported
        try:
            import coremltools as ct
        except ImportError as e:
            error_msg = f"Failed to import coremltools: {str(e)}"
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            return False
        
        # Load the model
        start_time = time.time()
        model = ct.models.MLModel(model_path)
        load_time = time.time() - start_time
        
        # Get model details
        spec = model.get_spec()
        
        # Store model metadata
        model_status["details"] = {
            "description": spec.description.metadata.shortDescription if hasattr(spec.description.metadata, "shortDescription") else "Unknown",
            "author": spec.description.metadata.author if hasattr(spec.description.metadata, "author") else "Unknown",
            "load_time_sec": load_time,
            "inputs": [input_desc.name for input_desc in spec.description.input],
            "outputs": [output_desc.name for output_desc in spec.description.output],
            "size_mb": os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0,
            "memory_saving_mode": memory_saving_mode
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model description: {model_status['details']['description']}")
        logger.info(f"Model inputs: {model_status['details']['inputs']}")
        logger.info(f"Model outputs: {model_status['details']['outputs']}")
        
        # Test model with a simple prediction
        test_success = _test_model_basic_prediction()
        if not test_success:
            error_msg = "Model loaded but failed basic prediction test"
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            model_status["loaded"] = False
            return False
        
        # Update status
        model_status["loaded"] = True
        model_status["last_error"] = None
        
        # Log the full success details including all paths
        logger.info(f"Model loaded from: {model_path}")
        if model_status.get("alternate_paths"):
            logger.info(f"Model also available at: {', '.join(model_status['alternate_paths'])}")
        
        return True
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_status["last_error"] = error_msg
        model_status["loaded"] = False
        return False

def _test_model_basic_prediction():
    """
    Test the model with a basic prediction to ensure it's working correctly.
    
    Returns:
        bool: True if prediction succeeds, False otherwise
    """
    global model
    
    if model is None:
        logger.error("Cannot test model - model not loaded")
        return False
    
    try:
        # Create a simple test input
        test_input = {
            'query_text': 'What is AI?',
            'passage_text': 'Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.'
        }
        
        # Try a prediction
        logger.info("Testing model with basic prediction")
        prediction = model.predict(test_input)
        
        # Check if prediction has expected format
        if isinstance(prediction, dict) and prediction:
            logger.info("Basic model prediction successful")
            return True
        else:
            logger.error(f"Unexpected prediction format: {type(prediction)}")
            return False
            
    except Exception as e:
        logger.error(f"Error during model test prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load the model on startup
model_loaded = load_model()

# Schedule periodic model checks
def schedule_model_checks():
    """Schedule periodic checks to ensure model is loaded"""
    import threading
    
    def check_model():
        global model, model_loaded, model_status
        
        # If model is not loaded, try to reload it
        if model is None or not model_status["loaded"]:
            logger.info("Scheduled check: Attempting to reload model")
            model_loaded = load_model()
        
        # Schedule the next check
        check_timer = threading.Timer(300, check_model)  # Check every 5 minutes
        check_timer.daemon = True
        check_timer.start()
    
    # Start the first check
    initial_timer = threading.Timer(60, check_model)  # First check after 1 minute
    initial_timer.daemon = True
    initial_timer.start()

# Start the scheduled checks
schedule_model_checks()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    web_search: bool = False
    search_query: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    intent: Optional[str] = None
    timestamp: Optional[str] = None

class ChatSession(BaseModel):
    messages: List[ChatMessage]
    session_id: str

@app.get("/")
async def root():
    """
    Enhanced health check endpoint with comprehensive model status information.
    This provides detailed diagnostics to help troubleshoot model loading issues
    across different deployment environments (Docker, Render, local development).
    """
    global model, model_loaded, model_status
    
    # Track execution time of the health check
    health_check_start = time.time()
    
    # 1. Get initialization information from app state
    init_info = getattr(app.state, 'init_result', {})
    
    # 2. Run a fresh model path check for latest information
    from .utils.model_utils import find_model_file, ensure_model_availability
    from .utils.initialization import detect_environment
    
    # Get detailed environment information
    env_info = detect_environment()
    
    # Check model locations
    model_found, current_path = find_model_file()
    
    # Update model status if paths have changed
    if current_path != model_status["path"] and model_found:
        logger.info(f"Model location changed from {model_status['path']} to {current_path}")
        model_status["path"] = current_path
        model_status["exists"] = model_found
    
    # Gather all model locations for comprehensive diagnostics
    all_model_locations = []
    
    # Get filesystem search results for the model
    model_files_found = []
    try:
        import subprocess
        if env_info.get("in_docker") or env_info.get("on_render"):
            # In container environments, use find command for broader search
            find_cmd = "find / -name 'BERTSQUADFP16.mlmodel' -type f 2>/dev/null | grep -v 'Permission denied'"
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                model_files_found = result.stdout.strip().split('\n')
        else:
            # In local development, search common directories
            search_dirs = [
                os.getcwd(),
                os.path.dirname(os.getcwd()),
                os.path.join(os.getcwd(), "backend"),
                os.path.join(os.getcwd(), "backend", "app"),
                os.path.join(os.getcwd(), "backend", "app", "model")
            ]
            for d in search_dirs:
                for root, _, files in os.walk(d):
                    if "BERTSQUADFP16.mlmodel" in files:
                        model_files_found.append(os.path.join(root, "BERTSQUADFP16.mlmodel"))
    except Exception as e:
        logger.error(f"Error searching for model files: {str(e)}")
    
    # Get all potential model locations
    all_potential_paths = []
    
    # 1. Add environment variable path
    model_data_path = os.environ.get('MODEL_DATA_PATH')
    if model_data_path:
        all_potential_paths.append(os.path.join(model_data_path, "BERTSQUADFP16.mlmodel"))
    
    # 2. Add search paths from environment
    if os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS'):
        for path in os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS', '').split(','):
            if path.strip():
                all_potential_paths.append(os.path.join(path.strip(), "BERTSQUADFP16.mlmodel"))
    
    # 3. Add standard locations
    app_dir = os.path.dirname(os.path.abspath(__file__))
    standard_paths = [
        os.path.join(app_dir, "model", "BERTSQUADFP16.mlmodel"),
        os.path.join(os.path.dirname(app_dir), "BERTSQUADFP16.mlmodel"),
        "/app/app/model/BERTSQUADFP16.mlmodel",
        "/tmp/model/BERTSQUADFP16.mlmodel",
        "/opt/render/project/src/backend/app/model/BERTSQUADFP16.mlmodel",
        "/opt/render/project/src/backend/BERTSQUADFP16.mlmodel"
    ]
    all_potential_paths.extend(standard_paths)
    
    # 4. Add paths found during filesystem search
    all_potential_paths.extend(model_files_found)
    
    # Remove duplicates while preserving order
    all_potential_paths = list(dict.fromkeys(all_potential_paths))
    
    # Check each path and add to diagnostics
    for path in all_potential_paths:
        try:
            exists = os.path.exists(path)
            size = 0
            is_valid = False
            access_error = None
            
            if exists:
                try:
                    size = os.path.getsize(path) / (1024 * 1024)
                    # Simple validation - check file size is reasonable
                    is_valid = size > 10  # Assume model is at least 10MB
                except Exception as e:
                    access_error = str(e)
            
            all_model_locations.append({
                "path": path,
                "exists": exists,
                "size_mb": round(size, 2) if exists else 0,
                "current": path == model_status["path"],
                "valid": is_valid,
                "error": access_error
            })
        except Exception as e:
            # Some paths might not be accessible
            all_model_locations.append({
                "path": path,
                "exists": False,
                "error": str(e)
            })
    
    # Try to reload the model if it's not loaded
    model_load_triggered = False
    if model is None and not model_status["loaded"] and not model_loaded:
        logger.info("Model not loaded, attempting quick load check")
        if model_found:
            model_load_triggered = True
            # Don't block health check with full model loading
            import threading
            
            def background_load():
                global model_loaded
                logger.info("Loading model in background thread from health check")
                model_loaded = load_model()
                logger.info(f"Background model load completed, success: {model_loaded}")
            
            thread = threading.Thread(target=background_load)
            thread.daemon = True
            thread.start()
    
    # Free Tier: Skip disk space checks to maintain compatibility
    # Just report a simple status message instead
    disk_space = {
        "note": "Disk space reporting disabled for Free Tier compatibility",
        "status": "Available disk space should be sufficient for model storage"
    }
    
    # Prepare comprehensive status response with actionable diagnostics
    response = {
        "status": "healthy" if model_status["loaded"] else "degraded",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_info.get("environment", settings.environment),
        "model": {
            "loaded": model_status["loaded"],
            "current_path": model_status["path"],
            "file_exists": model_status["exists"],
            "load_attempts": model_status["load_attempts"],
            "last_attempt": model_status["last_attempt_time"],
            "model_found": model_found,
            "alternative_locations": all_model_locations
        },
        "diagnostics": {
            "runtime_info": {
                "python_version": sys.version.split()[0],
                "coremltools_version": getattr(ct, "__version__", "unknown"),
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "memory_info": {
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2) if 'psutil' in sys.modules else "N/A",
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2) if 'psutil' in sys.modules else "N/A"
                }
            },
            "initialization": init_info,
            "deployment_info": {
                "in_docker": env_info.get("in_docker", False),
                "on_render": env_info.get("on_render", False),
                "model_data_path_env": model_data_path or "Not set",
                "search_paths_env": os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS', "Not set"),
                "hostname": env_info.get("hostname", "unknown"),
                "load_triggered": model_load_triggered
            },
            "disk_space": disk_space,
            "startup_time": app.state.startup_time if hasattr(app.state, 'startup_time') else "unknown",
            "health_check_time": round(time.time() - health_check_start, 3)
        },
        "api_version": settings.version
    }
    
    # Add more info if model is loaded
    if model_status["loaded"]:
        response["model"]["model_info"] = {
            "description": model_status["details"].get("description", "Unknown"),
            "load_time_sec": model_status["details"].get("load_time_sec", 0),
            "inputs": model_status["details"].get("inputs", []),
            "outputs": model_status["details"].get("outputs", [])
        }
    
    # If we have an error, include it with troubleshooting info
    if model_status["last_error"]:
        response["model"]["error"] = model_status["last_error"]
        response["model"]["error_help"] = model_status["last_error"].replace("/", "\/")
        
        # Add troubleshooting suggestions based on the error
        if "not found" in model_status["last_error"].lower() or "no such file" in model_status["last_error"].lower():
            # Model not found - comprehensive instructions
            response["diagnostics"]["action_plan"] = {
                "error_type": "MODEL_NOT_FOUND",
                "steps": [
                    {
                        "id": "check_model_exists",
                        "description": "Verify model file exists in repository",
                        "details": f"Found {len([loc for loc in all_model_locations if loc['exists']])} potential model files",
                        "command": "find / -name 'BERTSQUADFP16.mlmodel' -type f 2>/dev/null"
                    },
                    {
                        "id": "check_render_disk",
                        "description": "Check if persistent disk is mounted correctly on Render",
                        "details": "Ensure the mountPath in render.yaml is correctly set to /opt/render/project/src/backend/app/model",
                        "command": "ls -la /opt/render/project/src/backend/app/model/"
                    },
                    {
                        "id": "manually_download",
                        "description": "Try downloading the model directly",
                        "details": "Use the /api/download-model endpoint to force a model download",
                        "command": "curl -X POST https://yourdomain.com/api/download-model"
                    },
                    {
                        "id": "check_docker",
                        "description": "For Docker deployments, ensure volumes are correctly mounted",
                        "details": "Check docker-compose.yml volume mappings",
                        "command": "docker-compose config"
                    }
                ],
                "likely_cause": "The model file could not be found at any of the expected locations. "
                               + "This could be due to Git LFS issues, incorrect deployment configuration, "
                               + "or permissions problems."
            }
        elif "memory" in model_status["last_error"].lower():
            # Memory-related issues
            response["diagnostics"]["action_plan"] = {
                "error_type": "MEMORY_ERROR",
                "steps": [
                    {
                        "id": "check_resources",
                        "description": "Check available memory",
                        "details": "The model requires at least 2GB of free memory to load",
                        "command": "free -h"
                    },
                    {
                        "id": "increase_resources",
                        "description": "Increase memory allocation for the service",
                        "details": "On Render, upgrade to a plan with more memory"
                    }
                ],
                "likely_cause": "The server doesn't have enough memory to load the model. "
                               + "CoreML models can require significant memory resources."
            }
        else:
            # Generic model loading issues
            response["diagnostics"]["action_plan"] = {
                "error_type": "MODEL_LOADING_ERROR",
                "steps": [
                    {
                        "id": "verify_model",
                        "description": "Verify model integrity",
                        "details": "Check if the model file is corrupted",
                        "command": "cd backend && python verify_model.py"
                    },
                    {
                        "id": "check_coremltools",
                        "description": "Verify coremltools installation",
                        "details": "Make sure coremltools is properly installed",
                        "command": "pip install --upgrade coremltools==7.0"
                    },
                    {
                        "id": "restart_service",
                        "description": "Restart the service",
                        "details": "Try restarting the service to clear memory"
                    }
                ],
                "likely_cause": "There was an error loading the model with coremltools. "
                               + "This could be due to model corruption, compatibility issues, "
                               + "or resource constraints."
            }
    
    # Add manual download link if model is not found
    if not model_found:
        download_url = os.environ.get("MODEL_DOWNLOAD_URL", 
            "https://www.dropbox.com/scl/fi/w4iclrvil6vh39mg6j7pl/BERTSQUADFP16.mlmodel?rlkey=vbrr9jjvsam1xg9i4i19pkdra&st=ho9dyrm6&dl=1")
        response["diagnostics"]["download_info"] = {
            "manual_download_url": download_url,
            "api_download_endpoint": "/api/download-model",
            "target_location": model_status["path"]
        }
    
    # Include debug info for developers
    if settings.debug or settings.environment != "production":
        response["debug_info"] = {
            "python_path": sys.path,
            "loaded_modules": list(sys.modules.keys())[:20],  # First 20 modules
            "environment_variables": {k: v for k, v in os.environ.items() 
                                      if not any(secret in k.lower() for secret in ['key', 'secret', 'password', 'token'])}
        }
    
    return response


@app.post("/api/download-model")
async def download_model_endpoint():
    """Force download the model file from Dropbox."""
    try:
        from .utils.initialization import download_model_from_dropbox
        
        # Execute the download
        result = download_model_from_dropbox()
        
        if result["success"]:
            # Try to ensure the model is available in multiple locations
            from .utils.model_utils import ensure_model_availability
            ensure_result = ensure_model_availability()
            
            # Try to load the model
            global model_loaded
            model_loaded = load_model()
            
            return {
                "success": True,
                "message": f"Model downloaded successfully to {result['path']} ({result['size_mb']:.2f} MB)",
                "model_loaded": model_loaded,
                "download_details": result,
                "copies": ensure_result.get("copied_to", [])
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download model: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        logger.error(f"Error in download-model endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading model: {str(e)}"
        )

@app.post("/api/query", response_model=Dict[str, Any])
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def process_query(request: QueryRequest, request_obj: Request):
    """Process a query using the ML model"""
    global model, model_loaded
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request_obj)
    
    # Validate and sanitize the query
    sanitized_query = sanitize_input(request.query)
    is_valid, error_message = validate_query(sanitized_query)
    
    if not is_valid:
        logger.warning(f"Invalid query from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        logger.error("Model not loaded - cannot process query")
        raise HTTPException(status_code=503, detail="AI model not loaded. Please try again later.")
    
    # Track processing for performance metrics
    start_time = time.time()
    search_results = []
    
    try:
        # Determine if this is a code-related query
        query_category = categorize_query(sanitized_query)
        is_code_question = query_category.get("is_code_question", False)
        
        # Get context - either from request or web search
        context = request.context
        
        # If web search is enabled and no context is provided, fetch context from the web
        if request.web_search and settings.web_search.enabled and (not context or context.strip() == ""):
            search_query = request.search_query or sanitized_query
            logger.info(f"Performing web search for: {search_query}")
            context = await search_web(search_query)
            search_results = SEARCH_CACHE.get(search_query.lower().strip(), {}).get("results", [])
        
        if not context or context.strip() == "":
            context = settings.ai_response.default_context
        
        # Prepare input for the model
        model_input = {
            'query_text': sanitized_query,
            'passage_text': context
        }
        
        # Log the input
        logger.info(f"Model input: query_length={len(sanitized_query)}, context_length={len(context)}")
        
        # Set a timeout for the prediction
        try:
            # Get prediction from the model
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                prediction = model.predict(model_input)
        except asyncio.TimeoutError:
            logger.error(f"Model prediction timed out after {settings.model.predict_timeout_seconds} seconds")
            raise HTTPException(
                status_code=503, 
                detail=f"AI model is taking too long to respond. Please try a simpler query."
            )
        
        # Extract the answer from the prediction
        result = extract_answer(
            prediction, 
            context, 
            query=sanitized_query,
            include_confidence=True
        )
        
        # Determine intent from the query
        intent = determine_intent(sanitized_query)
        
        # Build the response
        response = {
            "answer": result["answer"],
            "intent": intent,
            "confidence": result["confidence"],
            "is_code_question": is_code_question,
            "context_used": True if (context and context != settings.ai_response.default_context) else False,
            "method": result["method_used"],
            "processing_time": time.time() - start_time
        }
        
        # Log the interaction for monitoring
        log_query_response(
            sanitized_query, 
            result["answer"], 
            metadata={
                "client_id": client_info["client_id"],
                "intent": intent,
                "confidence": result["confidence"],
                "method_used": result["method_used"],
                "processing_time": time.time() - start_time,
                "is_code_question": is_code_question,
                "context_used": bool(context and context != settings.ai_response.default_context)
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the failed interaction
        log_query_response(
            sanitized_query, 
            "ERROR", 
            metadata={
                "client_id": client_info["client_id"],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

@app.post("/api/chat/session", response_model=ChatSession)
async def create_chat_session():
    """Create a new chat session"""
    session_id = generate_session_id()
    return {"messages": [], "session_id": session_id}

@app.post("/api/chat/{session_id}", response_model=ChatMessage)
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def chat(session_id: str, message: ChatMessage, request: Request):
    """Add a message to a chat session and get a response"""
    global model, model_loaded
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request)
    logger.info(f"Chat request from client {client_info['client_id']} for session {session_id}")
    
    # Validate and sanitize the message content
    sanitized_content = sanitize_input(message.content)
    is_valid, error_message = validate_query(sanitized_content)
    
    if not is_valid:
        logger.warning(f"Invalid chat message from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Track processing for performance metrics
    start_time = time.time()
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        logger.error("Model not loaded - cannot process chat message")
        raise HTTPException(status_code=503, detail="AI model not loaded. Please try again later.")
    
    try:
        # Process the user message
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be sent")
        
        # Analyze the query to determine approach
        query_category = categorize_query(sanitized_content)
        is_code_query = query_category.get("is_code_question", False)
        
        # Determine intent - either from message or by analyzing content
        intent = message.intent if message.intent else determine_intent(sanitized_content)
        
        # Decide whether to use web search
        use_web_search = (
            settings.web_search.enabled and 
            settings.ai_response.enable_web_search_by_default and
            ("search" in intent.lower() or "web_search" in intent.lower())
        )
        
        # Get context from web if needed
        context = ""
        search_results = []
        
        if use_web_search:
            logger.info(f"Performing web search for chat: {sanitized_content[:50]}...")
            context = await search_web(sanitized_content)
            # Get the search results for potential citation
            search_query = sanitized_content.lower().strip()
            search_results = SEARCH_CACHE.get(search_query, {}).get("results", [])
        
        # If no context from search, use default
        if not context or context.strip() == "":
            context = settings.ai_response.default_context
            
            # For code questions, add a more specific context
            if is_code_query:
                lang = query_category.get("language")
                lang_context = f" I can help with {lang} code." if lang else ""
                context = f"I'll help you with your coding question.{lang_context} " + context
        
        # Prepare input for the model
        model_input = {
            'query_text': sanitized_content,
            'passage_text': context
        }
        
        # Log the model input for debugging
        logger.info(f"Chat model input: query_length={len(sanitized_content)}, context_length={len(context)}")
        
        # Set a timeout for the prediction
        try:
            # Get prediction from the model
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                prediction = model.predict(model_input)
        except asyncio.TimeoutError:
            logger.error(f"Model prediction timed out after {settings.model.predict_timeout_seconds} seconds")
            raise HTTPException(
                status_code=503, 
                detail=f"AI model is taking too long to respond. Please try a simpler message."
            )
        
        # Extract the answer from the prediction
        result = extract_answer(
            prediction, 
            context, 
            query=sanitized_content,
            include_confidence=True
        )
        
        # Create assistant response
        response = ChatMessage(
            role="assistant",
            content=result["answer"],
            intent=intent,
            timestamp=get_current_timestamp()
        )
        
        # Log the interaction for monitoring
        log_query_response(
            sanitized_content, 
            result["answer"], 
            metadata={
                "client_id": client_info["client_id"],
                "session_id": session_id,
                "intent": intent,
                "confidence": result["confidence"],
                "method_used": result["method_used"],
                "processing_time": time.time() - start_time,
                "is_code_question": is_code_query,
                "context_used": bool(context and context != settings.ai_response.default_context),
                "web_search_used": use_web_search
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the failed interaction
        log_query_response(
            sanitized_content if 'sanitized_content' in locals() else "ERROR", 
            "ERROR", 
            metadata={
                "client_id": client_info["client_id"],
                "session_id": session_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        
        # Return a user-friendly error message
        error_message = settings.ai_response.fallback_responses.get("error",
            "I encountered an error processing your message. Please try again with a different question.")
            
        return ChatMessage(
            role="assistant",
            content=error_message,
            intent="error",
            timestamp=get_current_timestamp()
        )

@app.get("/api/chat/{session_id}/export", response_model=Dict[str, Any])
async def export_chat_session(session_id: str):
    """Export a chat session as JSON"""
    # In a real implementation, you would retrieve the session from a database
    # For this example, we'll return a sample session
    sample_session = {
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how can you help me?",
                "intent": "greeting",
                "timestamp": "2023-11-01T12:00:00Z"
            },
            {
                "role": "assistant",
                "content": "Hi there! I'm Backdoor AI, and I'm here to help answer your questions and provide information. Feel free to ask me anything!",
                "intent": "greeting_response",
                "timestamp": "2023-11-01T12:00:05Z"
            }
        ]
    }
    
    return sample_session

# Helper functions
async def search_web(query: str) -> str:
    """Search the web for information related to the query"""
    try:
        async with aiohttp.ClientSession() as session:
            # Use a search engine API or scrape search results
            # For this example, we'll use a simple approach with DuckDuckGo
            search_url = f"https://duckduckgo.com/html/?q={query}"
            async with session.get(search_url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results
                    results = []
                    for result in soup.select(".result__body"):
                        title_elem = result.select_one(".result__title")
                        snippet_elem = result.select_one(".result__snippet")
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True)
                            results.append(f"{title}: {snippet}")
                    
                    return "\n\n".join(results) if results else "No relevant information found."
                else:
                    return "Unable to search the web at this time."
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return "Error occurred while searching the web."

from .utils.code_utils import contains_code, is_code_question, format_code_for_response
from .utils.response_utils import enhance_response, categorize_query, log_query_response
from .utils.search_utils import search_web
from .utils.security_utils import validate_query, sanitize_input, rate_limit_ip_and_tokens, setup_security, get_client_info
from .config import settings

def extract_answer(prediction: Dict[str, Any], context: str, query: str = "", include_confidence: bool = True) -> Dict[str, Any]:
    """
    Extract the answer from the model prediction and enhance it for the user.
    
    This function handles different output formats from the model:
    1. start_span/end_span indices for extractive QA
    2. Direct answer text fields
    3. Probability/confidence distributions for answer candidates
    
    Args:
        prediction: The model's prediction output
        context: The context text used for the question
        query: The original user query (for better response formatting)
        include_confidence: Whether to include confidence scores
    
    Returns:
        Dict with answer text, confidence level, and metadata
    """
    # Initialize the response structure
    response_data = {
        "answer": "",
        "confidence": 0.0,
        "method_used": "unknown",
        "context_used": bool(context and len(context) > 10),
        "metadata": {}
    }
    
    try:
        # Log the raw prediction type and structure
        logger.info(f"Raw prediction type: {type(prediction)}")
        
        # Create a normalized copy of the prediction for processing
        normalized_pred = {}
        
        # Convert numpy arrays to Python types for better handling
        if hasattr(prediction, 'tolist'):
            # If prediction is itself a numpy array
            prediction = prediction.tolist()
            response_data["metadata"]["prediction_type"] = "numpy_array"
        elif isinstance(prediction, dict):
            # If prediction contains numpy arrays as values
            response_data["metadata"]["prediction_type"] = "dict"
            response_data["metadata"]["prediction_keys"] = list(prediction.keys())
            
            for key, value in prediction.items():
                if hasattr(value, 'tolist'):
                    normalized_pred[key] = value.tolist()
                else:
                    normalized_pred[key] = value
        else:
            response_data["metadata"]["prediction_type"] = str(type(prediction))
        
        # Track if any method succeeded
        extraction_succeeded = False
        raw_answer = ""
        
        # APPROACH 1: Handle extractive QA output format (start_span/end_span)
        # This is the expected format for many BERT-based QA models
        if isinstance(prediction, dict) and 'start_span' in prediction and 'end_span' in prediction:
            try:
                # Handle array or single value
                start_idx = prediction['start_span'][0] if isinstance(prediction['start_span'], (list, tuple)) else prediction['start_span']
                end_idx = prediction['end_span'][0] if isinstance(prediction['end_span'], (list, tuple)) else prediction['end_span']
                
                # Convert to integers if they're numpy values or floats
                start_idx = int(float(start_idx))
                end_idx = int(float(end_idx))
                
                logger.info(f"Extracted indices: start={start_idx}, end={end_idx}, context_length={len(context)}")
                response_data["metadata"]["start_idx"] = start_idx
                response_data["metadata"]["end_idx"] = end_idx
                
                # Validate indices
                if 0 <= start_idx < len(context) and start_idx <= end_idx < len(context):
                    raw_answer = context[start_idx:end_idx+1].strip()
                    logger.info(f"Successfully extracted answer using span indices: '{raw_answer}'")
                    
                    # Get confidence score if available
                    if 'start_span_probs' in prediction and 'end_span_probs' in prediction:
                        start_prob = max(prediction['start_span_probs'])
                        end_prob = max(prediction['end_span_probs'])
                        confidence = (start_prob + end_prob) / 2
                        response_data["confidence"] = float(confidence)
                    else:
                        response_data["confidence"] = 0.8  # Default high confidence for span-based answers
                    
                    # Return the answer if it's not empty
                    if raw_answer:
                        extraction_succeeded = True
                        response_data["method_used"] = "span_indices"
                else:
                    logger.warning(f"Invalid indices: start={start_idx}, end={end_idx}, context_length={len(context)}")
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error processing start/end spans: {str(e)}")
        
        # APPROACH 2: Look for direct answer fields in the output
        if not extraction_succeeded and isinstance(prediction, dict):
            # Common field names for answers in different models
            answer_field_names = ['answer', 'text', 'response', 'output', 'answer_text', 'prediction']
            
            for field in answer_field_names:
                if field in prediction and isinstance(prediction[field], str) and prediction[field].strip():
                    raw_answer = prediction[field].strip()
                    logger.info(f"Found answer in '{field}' field: '{raw_answer}'")
                    extraction_succeeded = True
                    response_data["method_used"] = f"direct_field_{field}"
                    response_data["confidence"] = 0.9  # Direct fields are usually high confidence
                    break
        
        # APPROACH 3: If prediction is a string itself
        if not extraction_succeeded and isinstance(prediction, str) and prediction.strip():
            raw_answer = prediction.strip()
            logger.info(f"Prediction is a string: '{raw_answer}'")
            extraction_succeeded = True
            response_data["method_used"] = "string_prediction"
            response_data["confidence"] = 0.9
        
        # APPROACH 4: Use maximum probability token from start and end indices
        if not extraction_succeeded and 'start_span_probs' in prediction and 'end_span_probs' in prediction:
            try:
                # Try to find highest probability span
                start_probs = prediction['start_span_probs']
                end_probs = prediction['end_span_probs']
                
                # Get the top 3 most likely start and end positions
                start_indices = sorted(range(len(start_probs)), key=lambda i: -start_probs[i])[:3]
                end_indices = sorted(range(len(end_probs)), key=lambda i: -end_probs[i])[:3]
                
                best_score = 0
                best_answer = ""
                
                # Try different combinations
                for start_idx in start_indices:
                    for end_idx in end_indices:
                        if start_idx <= end_idx and end_idx < len(context):
                            answer = context[start_idx:end_idx+1].strip()
                            if answer and len(answer) > 2:  # Minimum answer length
                                score = (start_probs[start_idx] + end_probs[end_idx]) / 2
                                if score > best_score:
                                    best_score = score
                                    best_answer = answer
                
                if best_answer:
                    raw_answer = best_answer
                    logger.info(f"Extracted answer from probabilities: '{raw_answer}'")
                    extraction_succeeded = True
                    response_data["method_used"] = "probability_span"
                    response_data["confidence"] = float(best_score)
            except Exception as e:
                logger.warning(f"Error extracting answer from probabilities: {str(e)}")
        
        # FALLBACK: Generate a response based on the context
        if not extraction_succeeded and context and len(context) > 20:
            # Try to provide a meaningful response from the context
            context_start = context[:settings.ai_response.max_response_length//2].strip()
            if len(context) > settings.ai_response.max_response_length//2:
                context_start += "..."
                
            logger.info(f"Using context beginning as fallback")
            raw_answer = f"Based on the available information: {context_start}"
            extraction_succeeded = True
            response_data["method_used"] = "context_fallback"
            response_data["confidence"] = 0.3  # Low confidence for fallbacks
        
        # Final fallback if all else fails
        if not extraction_succeeded or not raw_answer:
            logger.warning("Could not extract a good answer from model prediction")
            
            # Check if it's a code question to provide a more specific fallback
            if query and is_code_question(query):
                raw_answer = settings.ai_response.fallback_responses.get("coding", 
                    "I'm not able to generate the code for this specific request. Could you provide more details?")
            else:
                raw_answer = settings.ai_response.fallback_responses.get("default",
                    "I'm unable to provide a specific answer based on the available information. Please try rephrasing your question.")
            
            response_data["method_used"] = "final_fallback"
            response_data["confidence"] = 0.1  # Very low confidence
        
        # Set the extracted answer
        response_data["answer"] = raw_answer
        
        # Enhance the response if we have the query
        if query:
            enhanced_answer = enhance_response(
                raw_answer, 
                query, 
                intent=determine_intent(query),
                confidence=response_data["confidence"] if include_confidence else None
            )
            response_data["answer"] = enhanced_answer
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log detailed error diagnostics
        logger.error(f"Prediction type: {type(prediction)}")
        logger.error(f"Context length: {len(context) if context else 0}")
        if isinstance(prediction, dict):
            logger.error(f"Prediction keys: {list(prediction.keys())}")
        
        # Return error response
        error_message = settings.ai_response.fallback_responses.get("error",
            "I encountered an error while processing your question. Please try again with a different phrasing.")
        
        response_data["answer"] = error_message
        response_data["method_used"] = "error_handler"
        response_data["confidence"] = 0.0
        
        return response_data

def determine_intent(query: str) -> str:
    """Determine the intent of the user's query"""
    query = query.lower()
    
    if any(word in query for word in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    elif any(word in query for word in ["search", "find", "look up", "google"]):
        return "web_search"
    elif any(word in query for word in ["what is", "who is", "explain", "define"]):
        return "definition"
    elif any(word in query for word in ["how to", "how do i"]):
        return "instruction"
    elif any(word in query for word in ["why", "reason"]):
        return "explanation"
    else:
        return "general_query"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

def get_current_timestamp() -> str:
    """Get the current timestamp in ISO format"""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
