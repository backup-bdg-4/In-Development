"""
Utilities for handling the CoreML model.
This module provides functionality to find, verify, and load the model
across different environments (development, Docker, Render, etc).
"""

import os
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
import shutil
from pathlib import Path
import coremltools as ct

logger = logging.getLogger(__name__)

def find_model_file(model_name: str = "BERTSQUADFP16.mlmodel") -> Tuple[bool, str]:
    """
    Find the model file by checking multiple possible locations.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Tuple of (found, path)
    """
    # Define all possible model locations in order of preference
    possible_locations = []
    
    # 1. Check environment variable (highest priority)
    model_data_path = os.environ.get('MODEL_DATA_PATH')
    if model_data_path:
        possible_locations.append(os.path.join(model_data_path, model_name))
    
    # 2. Check in app/model directory (standard location)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_locations.append(os.path.join(app_dir, "model", model_name))
    
    # 3. Check in backend directory (where GitHub Action places it)
    backend_dir = os.path.dirname(os.path.dirname(app_dir))
    possible_locations.append(os.path.join(backend_dir, model_name))
    
    # 4. Check for absolute /app paths (Docker container)
    possible_locations.append(f"/app/app/model/{model_name}")
    possible_locations.append(f"/app/{model_name}")
    
    # 5. Check the tmp directory (Render deployment)
    possible_locations.append(f"/tmp/model/{model_name}")
    
    # 6. Check for additional paths from environment variable
    if os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS'):
        search_paths = os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS', '').split(',')
        for path in search_paths:
            if path.strip():
                possible_locations.append(os.path.join(path.strip(), model_name))
    
    # Return the first location that exists
    for location in possible_locations:
        if os.path.exists(location):
            file_size_mb = os.path.getsize(location) / (1024 * 1024)
            logger.info(f"Found model at: {location} (Size: {file_size_mb:.2f} MB)")
            return True, location
    
    # If we get here, no existing locations were found
    # Return the preferred location (we'll try to get/download the model there)
    preferred_location = possible_locations[0] if possible_locations else os.path.join(
        app_dir, "model", model_name
    )
    logger.warning(f"No existing model found. Preferred path: {preferred_location}")
    return False, preferred_location

def validate_model(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the model is a valid CoreML model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (success, model_details)
    """
    try:
        start_time = time.time()
        
        # Try to load the model
        logger.info(f"Validating model at {model_path}")
        model = ct.models.MLModel(model_path)
        load_time = time.time() - start_time
        
        # Get basic model info to verify it loaded correctly
        spec = model.get_spec()
        
        # Extract model details
        model_details = {
            "description": spec.description.metadata.shortDescription if hasattr(spec.description.metadata, "shortDescription") else "Unknown",
            "author": spec.description.metadata.author if hasattr(spec.description.metadata, "author") else "Unknown",
            "load_time_sec": load_time,
            "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        }
        
        logger.info(f"Model validation successful: {model_details['description']}")
        return True, model_details
    
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False, {"error": str(e)}

def ensure_model_availability(model_name: str = "BERTSQUADFP16.mlmodel") -> Dict[str, Any]:
    """
    Ensure the model is available in at least one location.
    Attempts to copy between locations if needed.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Dictionary with status information
    """
    # 1. Find if model exists somewhere
    found, model_path = find_model_file(model_name)
    
    if not found:
        # Model not found anywhere, can't do much
        logger.error(f"Model {model_name} not found in any location")
        return {
            "success": False,
            "found": False,
            "error": f"Model {model_name} not found in any location",
            "path": model_path
        }
    
    # 2. Validate the model we found
    is_valid, details = validate_model(model_path)
    if not is_valid:
        return {
            "success": False,
            "found": True,
            "valid": False,
            "error": f"Model found but invalid: {details.get('error', 'Unknown error')}",
            "path": model_path
        }
    
    # 3. Copy to other directories for redundancy
    copied_locations = []
    
    # Define target directories to copy to
    target_dirs = []
    
    # Add MODEL_DATA_PATH if defined
    model_data_path = os.environ.get('MODEL_DATA_PATH')
    if model_data_path:
        os.makedirs(model_data_path, exist_ok=True)
        target_dirs.append(model_data_path)
    
    # Add /tmp/model directory
    tmp_model_dir = "/tmp/model"
    os.makedirs(tmp_model_dir, exist_ok=True)
    target_dirs.append(tmp_model_dir)
    
    # Add app/model directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_model_dir = os.path.join(app_dir, "model")
    os.makedirs(app_model_dir, exist_ok=True)
    target_dirs.append(app_model_dir)
    
    # Add Docker paths
    docker_app_model_dir = "/app/app/model"
    try:
        os.makedirs(docker_app_model_dir, exist_ok=True)
        target_dirs.append(docker_app_model_dir)
    except:
        # Might not be in Docker
        pass
    
    # Copy the model to all target directories
    for target_dir in target_dirs:
        target_path = os.path.join(target_dir, model_name)
        if os.path.abspath(target_path) != os.path.abspath(model_path):
            try:
                logger.info(f"Copying model from {model_path} to {target_path}")
                shutil.copy2(model_path, target_path)
                copied_locations.append(target_path)
            except Exception as e:
                logger.warning(f"Failed to copy model to {target_path}: {str(e)}")
    
    return {
        "success": True,
        "found": True,
        "valid": True,
        "source_path": model_path,
        "copied_to": copied_locations,
        "details": details
    }
