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
    
    Returns:
        Dict with environment information
    """
    env_info = {
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "in_docker": os.path.exists("/.dockerenv"),
        "on_render": os.environ.get("RUNNING_ON_RENDER") == "true",
        "python_version": sys.version,
        "model_data_path": os.environ.get("MODEL_DATA_PATH"),
        "cwd": os.getcwd(),
        "parent_dir": os.path.dirname(os.getcwd()),
        "user": os.environ.get("USER", "unknown"),
        "hostname": os.environ.get("HOSTNAME", "unknown"),
    }
    
    # Add render-specific information if running on render
    if env_info["on_render"]:
        env_info["render_service_id"] = os.environ.get("RENDER_SERVICE_ID")
        env_info["render_instance_id"] = os.environ.get("RENDER_INSTANCE_ID")
        env_info["render_git_commit"] = os.environ.get("RENDER_GIT_COMMIT")
        
    # Add Docker-specific information if running in Docker
    if env_info["in_docker"]:
        env_info["docker_hostname"] = os.environ.get("HOSTNAME")
    
    return env_info

def create_directories() -> Dict[str, bool]:
    """
    Create necessary directories for the application.
    
    Returns:
        Dict with directory creation results
    """
    results = {}
    
    # Get environment info
    env_info = detect_environment()
    
    # Define directories to create
    directories = [
        # App-specific directories
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"),
        
        # Check if MODEL_DATA_PATH is set
        env_info.get("model_data_path")
    ]
    
    # Add environment-specific directories
    if env_info["on_render"]:
        render_dirs = [
            "/opt/render/project/src/backend/app/model",
            "/tmp/model"
        ]
        directories.extend(render_dirs)
    
    if env_info["in_docker"]:
        docker_dirs = [
            "/app/app/model",
            "/app/backend/app/model",
            "/tmp/model"
        ]
        directories.extend(docker_dirs)
    
    # Create each directory
    for directory in directories:
        if directory:  # Skip None values
            try:
                os.makedirs(directory, exist_ok=True)
                results[directory] = True
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                results[directory] = False
                logger.error(f"Failed to create directory {directory}: {str(e)}")
    
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
    
    Args:
        dropbox_link: Dropbox direct download link
        output_path: Path to save the model file (if None, uses MODEL_DATA_PATH or app/model)
        
    Returns:
        Dict with status information
    """
    # Create directories first
    create_directories()
    
    # Determine output path if not provided
    if not output_path:
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
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Downloaded {file_size/(1024*1024):.2f} MB in {time.time() - start_time:.1f} seconds")
            
            # Copy to other important locations for redundancy
            try:
                from .model_utils import ensure_model_availability
                ensure_result = ensure_model_availability()
                logger.info(f"Model copied to {len(ensure_result.get('copied_to', []))} additional locations")
            except Exception as copy_error:
                logger.error(f"Error copying model to additional locations: {str(copy_error)}")
            
            return {
                "success": True,
                "path": output_path,
                "size_mb": file_size / (1024 * 1024),
                "download_time_sec": time.time() - start_time
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
    
    Returns:
        Dict with initialization status information
    """
    # Start time for tracking
    start_time = time.time()
    
    # 1. Detect environment
    env_info = detect_environment()
    logger.info(f"Detected environment: {env_info['environment']}")
    
    # 2. Create directories
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Log initialization result summary
    logger.info(f"Application initialization completed in {init_result['initialization_time_sec']:.2f} seconds")
    logger.info(f"Model found: {model_check['found']} at {model_check['path']}")
    
    return init_result
