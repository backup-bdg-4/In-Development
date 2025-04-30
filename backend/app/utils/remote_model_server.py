"""
Remote model server client for offloading model inference to an external service.
This module provides a way to connect to an external model server instead of
loading the model locally, reducing memory usage in the main application.
"""

import os
import logging
import json
import time
import requests
import aiohttp
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global variables
REMOTE_MODEL_SERVER_URL = os.environ.get('REMOTE_MODEL_SERVER_URL', 'https://your-model-server-url.com')
REMOTE_MODEL_SERVER_API_KEY = os.environ.get('REMOTE_MODEL_SERVER_API_KEY', '')
REMOTE_MODEL_SERVER_TIMEOUT = int(os.environ.get('REMOTE_MODEL_SERVER_TIMEOUT', '30'))
REMOTE_MODEL_SERVER_READY = False

def is_remote_server_available() -> bool:
    """Check if the remote model server is available."""
    try:
        response = requests.get(
            f"{REMOTE_MODEL_SERVER_URL}/health",
            headers=get_auth_headers(),
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking remote model server availability: {str(e)}")
        return False

def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for the remote model server."""
    headers = {
        "Content-Type": "application/json"
    }
    
    if REMOTE_MODEL_SERVER_API_KEY:
        headers["Authorization"] = f"Bearer {REMOTE_MODEL_SERVER_API_KEY}"
    
    return headers

def initialize_remote_model_server() -> bool:
    """
    Initialize the connection to the remote model server.
    
    Returns:
        True if initialization was successful, False otherwise
    """
    global REMOTE_MODEL_SERVER_READY
    
    try:
        # Check if the remote server is available
        if not is_remote_server_available():
            logger.error(f"Remote model server not available at {REMOTE_MODEL_SERVER_URL}")
            return False
        
        # Check if the model is loaded on the remote server
        response = requests.get(
            f"{REMOTE_MODEL_SERVER_URL}/model/status",
            headers=get_auth_headers(),
            timeout=REMOTE_MODEL_SERVER_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get model status from remote server: {response.text}")
            return False
        
        status = response.json()
        if not status.get("model_loaded", False):
            logger.error("Model not loaded on remote server")
            return False
        
        # Mark server as ready
        REMOTE_MODEL_SERVER_READY = True
        logger.info(f"Remote model server initialized successfully at {REMOTE_MODEL_SERVER_URL}")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing remote model server: {str(e)}")
        return False

def predict_with_remote_server(query: str, context: str) -> Dict[str, Any]:
    """
    Make a prediction using the remote model server.
    
    Args:
        query: Query text
        context: Context text
        
    Returns:
        Dictionary with prediction result
    """
    global REMOTE_MODEL_SERVER_READY
    
    if not REMOTE_MODEL_SERVER_READY:
        logger.error("Remote model server not ready")
        return {"error": "Remote model server not ready"}
    
    try:
        # Prepare the request payload
        payload = {
            "query_text": query,
            "passage_text": context
        }
        
        # Make the request to the remote server
        response = requests.post(
            f"{REMOTE_MODEL_SERVER_URL}/predict",
            headers=get_auth_headers(),
            json=payload,
            timeout=REMOTE_MODEL_SERVER_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Error from remote model server: {response.text}")
            return {"error": f"Remote model server error: {response.text}"}
        
        # Parse the response
        prediction_result = response.json()
        return prediction_result
    
    except Exception as e:
        logger.error(f"Error making prediction with remote server: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

async def predict_with_remote_server_async(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction using the remote model server with async support.
    
    Args:
        model_input: Dictionary with query_text and passage_text
        
    Returns:
        Dictionary with prediction result
    """
    global REMOTE_MODEL_SERVER_READY
    
    if not REMOTE_MODEL_SERVER_READY:
        logger.error("Remote model server not ready")
        return {"error": "Remote model server not ready"}
    
    # Extract query and context from model_input
    query = model_input.get('query_text', '')
    context = model_input.get('passage_text', '')
    
    try:
        # Prepare the request payload
        payload = {
            "query_text": query,
            "passage_text": context
        }
        
        # Make the request to the remote server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{REMOTE_MODEL_SERVER_URL}/predict",
                headers=get_auth_headers(),
                json=payload,
                timeout=REMOTE_MODEL_SERVER_TIMEOUT
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error from remote model server: {error_text}")
                    return {"error": f"Remote model server error: {error_text}"}
                
                # Parse the response
                prediction_result = await response.json()
                return prediction_result
    
    except Exception as e:
        logger.error(f"Error making async prediction with remote server: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

def check_remote_server_health() -> Dict[str, Any]:
    """
    Check the health of the remote model server.
    
    Returns:
        Dictionary with health status information
    """
    try:
        response = requests.get(
            f"{REMOTE_MODEL_SERVER_URL}/health",
            headers=get_auth_headers(),
            timeout=5
        )
        
        if response.status_code != 200:
            return {
                "status": "unhealthy",
                "error": f"Remote server returned status code {response.status_code}",
                "details": response.text
            }
        
        return {
            "status": "healthy",
            "details": response.json()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": None
        }

