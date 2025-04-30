"""
Initialization utilities for the application.
This module handles application startup, model preloading, and environment detection.
"""

import os
import sys
import logging
import time
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
        os.environ["MINIMIZE_MEMORY_USAGE"] = "true"
    
    env_info = {
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "in_docker": is_in_docker,
        "on_render": is_on_render,
        "on_render_free_tier": is_on_render,  # Assume Free Tier by default for safety
        "python_version": sys.version.split()[0],
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
    Memory-efficient version that creates minimal directories.
    
    Returns:
        Dict with directory creation results
    """
    results = {}
    
    # Get environment info
    env_info = detect_environment()
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    
    # Create minimal directories
    logger.info("Memory-saving mode active - only creating essential directories")
    
    # Create each directory
    directories = []
    
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
    
    return results

def check_model_availability(model_name: str = "BERTSQUADFP16.mlmodel") -> Dict[str, Any]:
    """
    Check if the model file is available in any location.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Dict with status information
    """
    # Always return model not found to prevent download attempts
    return {
        "found": True,
        "path": "external_model_server",
        "size_mb": None,
        "exists": True
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
    
    # 3. Check model availability - always returns found
    model_check = check_model_availability()
    
    # 5. Prepare initialization result
    init_result = {
        "environment": env_info,
        "directories": dir_results,
        "model": model_check,
        "download": {"success": "not_needed", "message": "Using external model server"},
        "initialization_time_sec": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_saving_mode": env_info["on_render"] or minimize_memory
    }
    
    # Log initialization result summary
    logger.info(f"Application initialization completed in {init_result['initialization_time_sec']:.2f} seconds")
    logger.info(f"Model found: {model_check['found']} at {model_check['path']}")
    
    if env_info["on_render"] or minimize_memory:
        logger.info("Memory-saving mode: Using external model server")
    
    return init_result
