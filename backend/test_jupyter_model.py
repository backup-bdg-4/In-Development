#!/usr/bin/env python3
"""
Test script for the Jupyter model server.
This script verifies that the Jupyter model server can be initialized and used for predictions.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

def test_jupyter_model_server():
    """Test the Jupyter model server functionality."""
    try:
        # Import the Jupyter model server module
        from app.utils.jupyter_model_server import (
            initialize_model_server,
            predict_with_jupyter,
            shutdown_jupyter_server,
            is_jupyter_server_running
        )
        
        logger.info("Jupyter model server module imported successfully")
        
        # Set the model path
        model_path = "/tmp/model/BERTSQUADFP16.mlmodel"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            logger.info("Checking for model in other locations...")
            
            # Check other possible locations
            possible_locations = [
                os.path.join(os.path.dirname(__file__), "app", "model", "BERTSQUADFP16.mlmodel"),
                os.path.join(os.path.dirname(__file__), "BERTSQUADFP16.mlmodel"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "BERTSQUADFP16.mlmodel")
            ]
            
            for loc in possible_locations:
                if os.path.exists(loc):
                    model_path = loc
                    logger.info(f"Found model at {model_path}")
                    
                    # Copy to /tmp/model for future use
                    os.makedirs("/tmp/model", exist_ok=True)
                    import shutil
                    shutil.copy2(loc, "/tmp/model/BERTSQUADFP16.mlmodel")
                    logger.info(f"Copied model to /tmp/model/BERTSQUADFP16.mlmodel")
                    model_path = "/tmp/model/BERTSQUADFP16.mlmodel"
                    break
            else:
                logger.error("Model not found in any location")
                return False
        
        # Initialize the model server
        logger.info(f"Initializing Jupyter model server with model at {model_path}")
        start_time = time.time()
        success = initialize_model_server(model_path)
        
        if not success:
            logger.error("Failed to initialize Jupyter model server")
            return False
        
        logger.info(f"Jupyter model server initialized in {time.time() - start_time:.2f} seconds")
        
        # Check if the server is running
        if not is_jupyter_server_running():
            logger.error("Jupyter model server is not running")
            return False
        
        logger.info("Jupyter model server is running")
        
        # Make a test prediction
        test_query = "What is AI?"
        test_context = "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines."
        
        logger.info(f"Making test prediction with query: '{test_query}'")
        prediction_result = predict_with_jupyter(test_query, test_context)
        
        if "error" in prediction_result:
            logger.error(f"Prediction error: {prediction_result['error']}")
            return False
        
        logger.info(f"Prediction result: {prediction_result}")
        
        # Shutdown the server
        logger.info("Shutting down Jupyter model server")
        shutdown_jupyter_server()
        
        logger.info("Test completed successfully")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import Jupyter model server module: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Error testing Jupyter model server: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Jupyter model server test")
    
    # Set environment variables for testing
    os.environ["USE_JUPYTER_MODEL_SERVER"] = "true"
    os.environ["MINIMIZE_MEMORY_USAGE"] = "true"
    
    # Run the test
    success = test_jupyter_model_server()
    
    if success:
        logger.info("✅ Jupyter model server test passed")
        sys.exit(0)
    else:
        logger.error("❌ Jupyter model server test failed")
        sys.exit(1)
