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
    Enhanced to specifically handle Render.com deployment paths.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Tuple of (found, path)
    """
    # Define all possible model locations in order of preference
    possible_locations = []
    search_paths_checked = []
    running_on_render = os.environ.get('RUNNING_ON_RENDER') == 'true'
    
    # 1. Check environment variable (highest priority)
    model_data_path = os.environ.get('MODEL_DATA_PATH')
    if model_data_path:
        model_path = os.path.join(model_data_path, model_name)
        possible_locations.append(model_path)
        search_paths_checked.append(model_path)
    
    # 2. Render.com specific paths (if running on Render)
    # Render uses a specific directory structure we need to account for
    if running_on_render:
        render_paths = [
            f"/opt/render/project/src/backend/app/model/{model_name}",  # Persistent disk path
            f"/opt/render/project/src/backend/{model_name}",            # Backend directory
            f"/opt/render/project/src/{model_name}",                    # Project root
            f"/opt/render/project/{model_name}",                        # Render directory
        ]
        possible_locations.extend(render_paths)
        search_paths_checked.extend(render_paths)
    
    # 3. Check in app/model directory (standard location)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_model_path = os.path.join(app_dir, "model", model_name)
    possible_locations.append(app_model_path)
    search_paths_checked.append(app_model_path)
    
    # 4. Check in backend directory (where GitHub Action places it)
    backend_dir = os.path.dirname(os.path.dirname(app_dir))
    backend_path = os.path.join(backend_dir, model_name)
    possible_locations.append(backend_path)
    search_paths_checked.append(backend_path)
    
    # 5. Check current working directory and parent for relative paths
    cwd = os.getcwd()
    possible_locations.append(os.path.join(cwd, model_name))
    possible_locations.append(os.path.join(cwd, "backend", model_name))
    possible_locations.append(os.path.join(cwd, "backend", "app", "model", model_name))
    possible_locations.append(os.path.join(os.path.dirname(cwd), model_name))
    search_paths_checked.extend([
        os.path.join(cwd, model_name),
        os.path.join(cwd, "backend", model_name),
        os.path.join(cwd, "backend", "app", "model", model_name),
        os.path.join(os.path.dirname(cwd), model_name)
    ])
    
    # 6. Check for absolute container paths (Docker container)
    docker_paths = [
        f"/app/app/model/{model_name}",
        f"/app/backend/app/model/{model_name}",
        f"/app/{model_name}",
        f"/app/backend/{model_name}"
    ]
    possible_locations.extend(docker_paths)
    search_paths_checked.extend(docker_paths)
    
    # 7. Check the tmp directory (often used in cloud deployments)
    tmp_paths = [
        f"/tmp/model/{model_name}",
        f"/tmp/{model_name}"
    ]
    possible_locations.extend(tmp_paths)
    search_paths_checked.extend(tmp_paths)
    
    # 8. Check for additional paths from environment variable
    if os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS'):
        env_paths = os.environ.get('BACKDOOR_MODEL_SEARCH_PATHS', '').split(',')
        for path in env_paths:
            if path.strip():
                path_with_model = os.path.join(path.strip(), model_name)
                possible_locations.append(path_with_model)
                search_paths_checked.append(path_with_model)
    
    # Log all paths we're checking
    logger.info(f"Searching for model in {len(possible_locations)} possible locations")
    
    # Return the first location that exists
    for location in possible_locations:
        try:
            if os.path.exists(location):
                file_size_mb = os.path.getsize(location) / (1024 * 1024)
                logger.info(f"✅ Found model at: {location} (Size: {file_size_mb:.2f} MB)")
                return True, location
        except Exception as e:
            # Some paths might not be accessible
            logger.warning(f"Error checking path {location}: {str(e)}")
    
    # If we get here, no existing locations were found
    # Log all paths we checked for easier debugging
    logger.warning(f"❌ Model not found in any of these locations:")
    for path in search_paths_checked:
        logger.warning(f" - {path}")
    
    # Return the preferred location (we'll try to get/download the model there)
    preferred_location = os.environ.get('MODEL_DATA_PATH', None)
    
    # If running on Render, use its persistent disk
    if running_on_render and preferred_location is None:
        preferred_location = "/opt/render/project/src/backend/app/model"
    
    # Fallback to app directory if no environment variable
    if preferred_location is None:
        preferred_location = os.path.join(app_dir, "model")
    
    # Add model name to the path
    preferred_location = os.path.join(preferred_location, model_name)
    
    logger.warning(f"No existing model found. Will use path: {preferred_location}")
    return False, preferred_location

def validate_model(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the model is a valid CoreML model.
    Memory-efficient version that avoids loading the full model in memory-saving mode.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (success, model_details)
    """
    try:
        start_time = time.time()
        minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
        
        # Check if file exists and has reasonable size
        if not os.path.exists(model_path):
            return False, {"error": f"Model file not found at {model_path}"}
        
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # In memory-saving mode, just check file existence and size
        if minimize_memory:
            logger.info(f"Memory-saving mode: Skipping full model validation for {model_path}")
            logger.info(f"File exists with size: {file_size_mb:.2f} MB")
            
            # Basic validation - check file size is reasonable (> 100MB for BERT model)
            if file_size_mb < 100:
                logger.warning(f"Model file size ({file_size_mb:.2f} MB) seems too small for a BERT model")
                
            model_details = {
                "description": "Validation skipped in memory-saving mode",
                "author": "Unknown",
                "load_time_sec": time.time() - start_time,
                "size_mb": file_size_mb,
                "validation_skipped": True
            }
            
            return True, model_details
        
        # Standard validation - load the model
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
            "size_mb": file_size_mb,
        }
        
        logger.info(f"Model validation successful: {model_details['description']}")
        
        # Clear model from memory
        model = None
        
        return True, model_details
    
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False, {"error": str(e)}

def ensure_model_availability(model_name: str = "BERTSQUADFP16.mlmodel") -> Dict[str, Any]:
    """
    Ensure the model is available in at least one location.
    Memory-efficient version that minimizes copies for Render Free Tier.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Dictionary with status information
    """
    # Track starting time for performance monitoring
    start_time = time.time()
    running_on_render = os.environ.get('RUNNING_ON_RENDER') == 'true'
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    
    # 1. Find if model exists somewhere
    found, model_path = find_model_file(model_name)
    
    if not found:
        # Model not found anywhere, log additional diagnostic info
        logger.error(f"⚠️ Model {model_name} not found in any location")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir('.')}")
        
        # Check if we're running in a container by looking for .dockerenv file
        in_container = os.path.exists('/.dockerenv')
        logger.error(f"Running in container: {in_container}")
        
        # Look for the model file anywhere in the filesystem (limited search)
        try:
            import subprocess
            logger.error("Attempting to search for model file across filesystem...")
            find_cmd = f"find / -name '{model_name}' -type f 2>/dev/null | grep -v 'Permission denied'"
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                logger.error(f"Found model files at: {result.stdout}")
                
                # Try to use one of the found files
                found_paths = result.stdout.strip().split('\n')
                if found_paths:
                    model_path = found_paths[0]
                    logger.error(f"Will attempt to use found model at: {model_path}")
                    found = True
        except Exception as search_err:
            logger.error(f"Error during filesystem search: {search_err}")
        
        if not found:
            return {
                "success": False,
                "found": False,
                "error": f"Model {model_name} not found in any location",
                "path": model_path,
                "search_time_sec": time.time() - start_time
            }
    
    # 2. Validate the model we found
    logger.info(f"Validating model at {model_path}")
    is_valid, details = validate_model(model_path)
    if not is_valid:
        return {
            "success": False,
            "found": True,
            "valid": False,
            "error": f"Model found but invalid: {details.get('error', 'Unknown error')}",
            "path": model_path,
            "search_time_sec": time.time() - start_time
        }
    
    # 3. Copy to other directories for redundancy - but only if not in memory-saving mode
    copied_locations = []
    copy_errors = []
    
    # If we're on Render Free Tier or memory-saving mode is enabled,
    # only ensure the model is in /tmp/model and skip other copies
    if running_on_render or minimize_memory:
        logger.info("Running in memory-saving mode - minimizing model copies")
        
        # Only ensure the model is in /tmp/model
        tmp_model_dir = "/tmp/model"
        try:
            os.makedirs(tmp_model_dir, exist_ok=True)
            tmp_model_path = os.path.join(tmp_model_dir, model_name)
            
            # Only copy if needed and not already in /tmp/model
            if os.path.abspath(tmp_model_path) != os.path.abspath(model_path):
                if os.path.exists(tmp_model_path):
                    # Check if sizes match
                    source_size = os.path.getsize(model_path)
                    target_size = os.path.getsize(tmp_model_path)
                    
                    if source_size == target_size:
                        logger.info(f"Model already exists at {tmp_model_path} with matching size")
                        copied_locations.append(tmp_model_path)
                    else:
                        # Only copy if sizes don't match
                        logger.info(f"Copying model to {tmp_model_path} (size mismatch)")
                        shutil.copy2(model_path, tmp_model_path)
                        copied_locations.append(tmp_model_path)
                else:
                    # Copy if doesn't exist
                    logger.info(f"Copying model to {tmp_model_path}")
                    shutil.copy2(model_path, tmp_model_path)
                    copied_locations.append(tmp_model_path)
            else:
                # Model is already in /tmp/model
                logger.info(f"Model already in primary location {tmp_model_path}")
                copied_locations.append(tmp_model_path)
                
            # Set MODEL_DATA_PATH to /tmp/model
            os.environ["MODEL_DATA_PATH"] = "/tmp/model"
            
            # If the model is not in /tmp/model, update the model_path
            if os.path.abspath(model_path) != os.path.abspath(tmp_model_path) and os.path.exists(tmp_model_path):
                model_path = tmp_model_path
                logger.info(f"Updated primary model path to {model_path}")
        except Exception as e:
            error_msg = f"Failed to ensure model in /tmp/model: {str(e)}"
            logger.warning(error_msg)
            copy_errors.append(error_msg)
    else:
        # Standard behavior for non-Render environments - copy to multiple locations
        # Define target directories to copy to
        target_dirs = []
        
        # Add MODEL_DATA_PATH if defined
        model_data_path = os.environ.get('MODEL_DATA_PATH')
        if model_data_path:
            try:
                os.makedirs(model_data_path, exist_ok=True)
                target_dirs.append(model_data_path)
            except Exception as e:
                copy_errors.append(f"Failed to create MODEL_DATA_PATH directory {model_data_path}: {str(e)}")
        
        # Add Render.com specific paths if running on Render
        if running_on_render:
            render_paths = [
                "/opt/render/project/src/backend/app/model",  # Persistent disk path
                "/opt/render/project/src/backend",           # Backend directory
            ]
            for path in render_paths:
                try:
                    os.makedirs(path, exist_ok=True)
                    target_dirs.append(path)
                except Exception as e:
                    copy_errors.append(f"Failed to create Render directory {path}: {str(e)}")
        
        # Add /tmp/model directory
        tmp_model_dir = "/tmp/model"
        try:
            os.makedirs(tmp_model_dir, exist_ok=True)
            target_dirs.append(tmp_model_dir)
        except Exception as e:
            copy_errors.append(f"Failed to create temp directory {tmp_model_dir}: {str(e)}")
        
        # Add app/model directory
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_model_dir = os.path.join(app_dir, "model")
        try:
            os.makedirs(app_model_dir, exist_ok=True)
            target_dirs.append(app_model_dir)
        except Exception as e:
            copy_errors.append(f"Failed to create app model directory {app_model_dir}: {str(e)}")
        
        # Add Docker paths
        docker_paths = [
            "/app/app/model", 
            "/app/backend/app/model"
        ]
        for docker_path in docker_paths:
            try:
                os.makedirs(docker_path, exist_ok=True)
                target_dirs.append(docker_path)
            except:
                # Might not be in Docker or permissions issue
                pass
        
        # Add current directory and backend/app/model for local development
        try:
            cwd = os.getcwd()
            local_dirs = [
                os.path.join(cwd, "backend", "app", "model"),
                os.path.join(cwd, "app", "model")
            ]
            for local_dir in local_dirs:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    target_dirs.append(local_dir)
                except Exception as e:
                    copy_errors.append(f"Failed to create local directory {local_dir}: {str(e)}")
        except Exception as e:
            copy_errors.append(f"Error setting up local directories: {str(e)}")
        
        # Remove duplicates while preserving order
        target_dirs = list(dict.fromkeys(target_dirs))
        
        logger.info(f"Copying model to {len(target_dirs)} target directories for redundancy")
        
        # Copy the model to all target directories
        for target_dir in target_dirs:
            target_path = os.path.join(target_dir, model_name)
            if os.path.abspath(target_path) != os.path.abspath(model_path):
                try:
                    # Check if target file already exists and has the same size
                    if os.path.exists(target_path):
                        source_size = os.path.getsize(model_path)
                        target_size = os.path.getsize(target_path)
                        
                        if source_size == target_size:
                            logger.info(f"Model already exists at {target_path} with matching size ({source_size} bytes)")
                            copied_locations.append(target_path)
                            continue
                        else:
                            logger.warning(f"Size mismatch for existing model at {target_path}: {source_size} vs {target_size} bytes")
                    
                    # Copy the file
                    logger.info(f"Copying model from {model_path} to {target_path}")
                    shutil.copy2(model_path, target_path)
                    
                    # Verify the copy succeeded
                    if os.path.exists(target_path):
                        logger.info(f"✅ Successfully copied model to {target_path}")
                        copied_locations.append(target_path)
                    else:
                        error_msg = f"Copy operation didn't fail but file doesn't exist at {target_path}"
                        logger.warning(error_msg)
                        copy_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Failed to copy model to {target_path}: {str(e)}"
                    logger.warning(error_msg)
                    copy_errors.append(error_msg)
    
    # Calculate total process time
    process_time = time.time() - start_time
    
    # Get a directory listing where the model should be for logging
    directory_listings = {}
    for dir_path in list(set([os.path.dirname(p) for p in copied_locations + [model_path]])):
        try:
            if os.path.exists(dir_path):
                directory_listings[dir_path] = os.listdir(dir_path)
        except Exception:
            pass
    
    return {
        "success": True,
        "found": True,
        "valid": True,
        "source_path": model_path,
        "copied_to": copied_locations,
        "details": details,
        "copy_errors": copy_errors if copy_errors else None,
        "directory_listings": directory_listings,
        "process_time_sec": process_time,
        "memory_saving_mode": running_on_render or minimize_memory
    }
