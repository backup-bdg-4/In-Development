"""
External model server interface for memory-efficient model loading.
This module provides a way to connect to an external model server
instead of loading the model locally.
"""

import os
import sys
import logging
import json
import time
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global variables
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:5000")
MODEL_SERVER_READY = False

def is_model_server_running() -> bool:
    """Check if the external model server is running."""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"External model server health check failed: {str(e)}")
        return False

def initialize_model_server(model_path: str = None) -> bool:
    """
    Initialize connection to the external model server.
    
    Args:
        model_path: Not used, kept for compatibility
        
    Returns:
        True if server is running, False otherwise
    """
    global MODEL_SERVER_READY
    
    # Check if server is running
    server_running = is_model_server_running()
    
    if server_running:
        logger.info(f"Connected to external model server at {MODEL_SERVER_URL}")
        MODEL_SERVER_READY = True
        return True
    else:
        logger.warning(f"External model server not available at {MODEL_SERVER_URL}")
        MODEL_SERVER_READY = False
        return False

def predict_with_external_server(query: str, context: str) -> Dict[str, Any]:
    """
    Make a prediction using the external model server.
    
    Args:
        query: Query text
        context: Context text
        
    Returns:
        Dictionary with prediction result
    """
    global MODEL_SERVER_READY
    
    if not MODEL_SERVER_READY:
        # Try to initialize connection
        if not initialize_model_server():
            logger.error("External model server not available")
            return {"error": "External model server not available"}
    
    try:
        # Prepare request data
        request_data = {
            "query_text": query,
            "passage_text": context
        }
        
        # Send request to external server
        response = requests.post(
            f"{MODEL_SERVER_URL}/predict",
            json=request_data,
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"External model server returned error: {response.status_code} - {response.text}")
            return {"error": f"External model server error: {response.status_code}"}
    
    except Exception as e:
        logger.error(f"Error making prediction with external server: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

async def predict_with_external_server_async(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction using the external model server with async support.
    
    Args:
        model_input: Dictionary with query_text and passage_text
        
    Returns:
        Dictionary with prediction result
    """
    global MODEL_SERVER_READY
    
    if not MODEL_SERVER_READY:
        # Try to initialize connection
        if not initialize_model_server():
            logger.error("External model server not available")
            return {"error": "External model server not available"}
    
    # Extract query and context from model_input
    query = model_input.get('query_text', '')
    context = model_input.get('passage_text', '')
    
    # Use the synchronous function for now
    # In a real implementation, this would use aiohttp for async requests
    return predict_with_external_server(query, context)

def shutdown_model_server():
    """Shutdown connection to the external model server."""
    global MODEL_SERVER_READY
    
    logger.info("Disconnecting from external model server")
    MODEL_SERVER_READY = False

