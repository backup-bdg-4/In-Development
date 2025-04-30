"""
Initialization utilities for the application.
This module handles application startup, model preloading, and environment detection.
"""

import os
import sys
import logging
import time
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def detect_environment() -> Dict[str, Any]:
    """
    Detect the current runtime environment.
    Free Tier compatible version that prioritizes /tmp storage.
    
    Returns:
        Dict with environment information
    """
    # Detect if we're on Render
    is_on_render = os.environ.get("RUNNING_ON_RENDER") == "true" or "render" in os.environ.get("HOSTNAME", "")
    is_in_docker = os.path.exists("/.dockerenv")
    
    # If on Render Free Tier, prefer /tmp/model path since it's most reliable
    # and set a flag to avoid multiple model copies
    if is_on_render:
        os.environ["MODEL_DATA_PATH"] = "/tmp/model"
        os.environ["RENDER_FREE_TIER"] = "true"
        os.environ["MINIMIZE_MEMORY_USAGE"] = "true"
    
    env_info = {
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "in_docker": is_in_docker,
        "on_render": is_on_render,
        "on_render_free_tier": is_on_render,  # Assume Free Tier by default for safety
        "python_version": sys.version.split()[0],
        "model_data_path": os.environ.get("MODEL_DATA_PATH"),
        "cwd": os.getcwd(),
        "parent_dir": os.path.dirname(os.getcwd()),
        "user": os.environ.get("USER", "unknown"),
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "minimize_memory": os.environ.get("MINIMIZE_MEMORY_USAGE") == "true"
    }
    
    # Add minimal Render-specific information for diagnostics
    if env_info["on_render"]:
        env_info["render_service_id"] = os.environ.get("RENDER_SERVICE_ID", "unknown")
        
    return env_info

def create_directories() -> Dict[str, bool]:
    """
    Create necessary directories for the application.
    Memory-efficient version that prioritizes /tmp/model directory.
    
    Returns:
        Dict with directory creation results
    """
    results = {}
    
    # Get environment info
    env_info = detect_environment()
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    
    # For Render Free Tier or memory-saving mode, only create /tmp/model
    if env_info["on_render"] or minimize_memory:
        logger.info("Memory-saving mode active - only creating essential directories")
        # Always set MODEL_DATA_PATH to /tmp/model
        os.environ["MODEL_DATA_PATH"] = "/tmp/model"
        
        # Only create /tmp/model in memory-saving mode
        directories = ["/tmp/model"]
    else:
        # Standard mode - create multiple directories
        directories = [
            # Temp directory
            "/tmp/model",
            
            # App-specific directories
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"),
            
            # Check if MODEL_DATA_PATH is set and different from /tmp/model
            env_info.get("model_data_path") if env_info.get("model_data_path") != "/tmp/model" else None
        ]
        
        # Add basic Render paths
        if env_info["on_render"]:
            render_dirs = [
                # Current directory models
                os.path.join(os.getcwd(), "backend", "app", "model"),
                os.path.join(os.getcwd(), "backend"),
                # Standard Render paths
                "/opt/render/project/src/backend/app/model"
            ]
            directories.extend(render_dirs)
        
        if env_info["in_docker"]:
            docker_dirs = [
                "/app/app/model",
                "/app/backend/app/model"
            ]
            directories.extend(docker_dirs)
    
    # Remove duplicates and None values
    directories = [d for d in directories if d]
    directories = list(dict.fromkeys(directories))
    
    # Create each directory
    logger.info(f"Creating {len(directories)} directories for model storage")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            results[directory] = True
            logger.info(f"✓ Created directory: {directory}")
        except Exception as e:
            results[directory] = False
            logger.warning(f"✗ Failed to create directory {directory}: {str(e)}")
    
    # Make sure /tmp/model exists and is writable
    if "/tmp/model" in results and results["/tmp/model"]:
        try:
            # Test write access
            test_file = os.path.join("/tmp/model", ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logger.info("✓ /tmp/model is writable (good for Free Tier)")
        except Exception as e:
            logger.warning(f"✗ /tmp/model exists but is not writable: {str(e)}")
    
    return results

def check_model_availability(model_name: str = "BERTSQUADFP16.mlmodel") -> Dict[str, Any]:
    """
    Check if the model file is available in any location.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Dict with status information
    """
    from .model_utils import find_model_file
    
    found, model_path = find_model_file(model_name)
    
    # Check file size if found
    file_size_mb = None
    if found and os.path.exists(model_path):
        try:
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting file size for {model_path}: {str(e)}")
    
    return {
        "found": found,
        "path": model_path,
        "size_mb": file_size_mb,
        "exists": found and os.path.exists(model_path)
    }

def download_model_from_dropbox(
    dropbox_link: str = "https://www.dropbox.com/scl/fi/w4iclrvil6vh39mg6j7pl/BERTSQUADFP16.mlmodel?rlkey=vbrr9jjvsam1xg9i4i19pkdra&st=ho9dyrm6&dl=1", 
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download the model file directly from Dropbox.
    Memory-efficient version that prioritizes /tmp/model and avoids redundant copies.
    
    Args:
        dropbox_link: Dropbox direct download link
        output_path: Path to save the model file (if None, uses /tmp/model for Free Tier)
        
    Returns:
        Dict with status information
    """
    # Create directories first
    create_directories()
    
    # Get environment info
    env_info = detect_environment()
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    
    # For Render Free Tier or memory-saving mode, always use /tmp/model
    if env_info["on_render"] or minimize_memory:
        output_path = "/tmp/model/BERTSQUADFP16.mlmodel"
        logger.info("Memory-saving mode active - downloading directly to /tmp/model")
    elif not output_path:
        # Use MODEL_DATA_PATH if set
        model_data_path = os.environ.get('MODEL_DATA_PATH')
        if model_data_path:
            output_path = os.path.join(model_data_path, "BERTSQUADFP16.mlmodel")
        else:
            # Use app/model directory as fallback
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(app_dir, "model", "BERTSQUADFP16.mlmodel")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Track time for the operation
    start_time = time.time()
    
    try:
        logger.info(f"Downloading model from Dropbox to {output_path}")
        
        # Use stream=True to handle large files
        with requests.get(dropbox_link, stream=True, timeout=60) as response:
            response.raise_for_status()
            
            # Get content length if available
            total_size = int(response.headers.get('content-length', 0))
            total_size_mb = total_size / (1024 * 1024) if total_size else 0
            
            logger.info(f"Model file size: {total_size_mb:.2f} MB")
            
            # Write the file
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress periodically
                        if total_size and downloaded % (10 * 1024 * 1024) < 8192:  # Log every 10MB
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size_mb:.1f} MB)")
        
        # Verify the file was downloaded successfully
        download_succeeded = os.path.exists(output_path)
        
        if download_succeeded:
            file_size = os.path.getsize(output_path)
            logger.info(f"Downloaded {file_size/(1024*1024):.2f} MB in {time.time() - start_time:.1f} seconds")
            
            # In memory-saving mode, don't create redundant copies
            copied_locations = []
            
            # Only create a copy in /tmp/model if we're not already there and not in memory-saving mode
            if not (env_info["on_render"] or minimize_memory) and output_path != "/tmp/model/BERTSQUADFP16.mlmodel":
                try:
                    tmp_path = "/tmp/model/BERTSQUADFP16.mlmodel"
                    logger.info(f"Creating copy in {tmp_path}")
                    os.makedirs("/tmp/model", exist_ok=True)
                    shutil.copy2(output_path, tmp_path)
                    copied_locations.append(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to create copy in /tmp/model: {str(e)}")
            
            # Return success with copied locations
            return {
                "success": True,
                "path": output_path,
                "size_mb": file_size / (1024 * 1024),
                "download_time_sec": time.time() - start_time,
                "copies": copied_locations,
                "memory_saving_mode": env_info["on_render"] or minimize_memory
            }
        else:
            return {
                "success": False,
                "error": "File download appeared to succeed but file doesn't exist",
                "path": output_path,
                "download_time_sec": time.time() - start_time
            }
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "path": output_path,
            "download_time_sec": time.time() - start_time
        }

def initialize_app() -> Dict[str, Any]:
    """
    Initialize the application.
    Memory-efficient version that minimizes redundant operations.
    
    Returns:
        Dict with initialization status information
    """
    # Start time for tracking
    start_time = time.time()
    
    # 1. Detect environment
    env_info = detect_environment()
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    logger.info(f"Detected environment: {env_info['environment']}")
    
    if env_info["on_render"] or minimize_memory:
        logger.info("Memory-saving mode active - optimizing for minimal memory usage")
    
    # 2. Create directories (will be minimal in memory-saving mode)
    dir_results = create_directories()
    
    # 3. Check model availability
    model_check = check_model_availability()
    
    # 4. Download model if not available
    if not model_check["found"]:
        logger.warning("Model not found, attempting direct download from Dropbox")
        download_result = download_model_from_dropbox()
        model_check = check_model_availability()
    else:
        download_result = {"success": "not_needed", "message": "Model already available"}
    
    # 5. Prepare initialization result
    init_result = {
        "environment": env_info,
        "directories": dir_results,
        "model": model_check,
        "download": download_result,
        "initialization_time_sec": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_saving_mode": env_info["on_render"] or minimize_memory
    }
    
    # Log initialization result summary
    logger.info(f"Application initialization completed in {init_result['initialization_time_sec']:.2f} seconds")
    logger.info(f"Model found: {model_check['found']} at {model_check['path']}")
    
    if env_info["on_render"] or minimize_memory:
        logger.info("Memory-saving mode: Using /tmp/model as primary storage location")
        logger.info("Memory-saving mode: Minimizing redundant model copies")
    
    return init_result
