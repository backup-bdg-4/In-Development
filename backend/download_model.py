import os
import logging
import json
import traceback
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model information
MODEL_INFO = {
    "name": "BERTSQUADFP16",
    "type": "BERT-based question answering model",
    "expected_inputs": ["query_text", "passage_text"],
    "expected_outputs": ["start_span", "end_span"]
}

# Define model paths
def get_model_path() -> str:
    """
    Get the path to the model file by checking multiple possible locations.
    This function handles different deployment environments (local dev, Docker, Render, etc.).
    
    Returns:
        str: Path to the model file (preferred location if multiple exist)
    """
    # Define all possible model locations in order of preference
    possible_locations = []
    
    # 1. Check environment variable (highest priority)
    model_data_path = os.environ.get('MODEL_DATA_PATH', None)
    if model_data_path:
        possible_locations.append(os.path.join(model_data_path, "BERTSQUADFP16.mlmodel"))
    
    # 2. Check in app/model directory (standard location)
    possible_locations.append(os.path.join(os.path.dirname(__file__), "app", "model", "BERTSQUADFP16.mlmodel"))
    
    # 3. Check in backend directory (where GitHub Action places it)
    possible_locations.append(os.path.join(os.path.dirname(__file__), "BERTSQUADFP16.mlmodel"))
    
    # 4. Check for absolute /app paths (Docker container)
    possible_locations.append("/app/app/model/BERTSQUADFP16.mlmodel")
    possible_locations.append("/app/BERTSQUADFP16.mlmodel")
    
    # 5. Check the tmp directory (Render deployment)
    possible_locations.append("/tmp/model/BERTSQUADFP16.mlmodel")
    
    # Return the first location that exists
    for location in possible_locations:
        if os.path.exists(location):
            logger.info(f"Found model at: {location}")
            return location
    
    # If we get here, no existing locations were found
    # Return the preferred location (we'll try to get/download the model there)
    preferred_location = possible_locations[0] if possible_locations else os.path.join(
        os.path.dirname(__file__), "app", "model", "BERTSQUADFP16.mlmodel"
    )
    logger.info(f"No existing model found, will use path: {preferred_location}")
    return preferred_location

def validate_model(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the model is a valid CoreML model.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (success, model_details)
    """
    try:
        import coremltools as ct
        # Try to load the model to validate it
        model = ct.models.MLModel(model_path)
        
        # Get basic model info to verify it loaded correctly
        spec = model.get_spec()
        
        # Collect model details for diagnostics
        model_details = {
            "description": spec.description.metadata.shortDescription if hasattr(spec.description.metadata, "shortDescription") else "Unknown",
            "author": spec.description.metadata.author if hasattr(spec.description.metadata, "author") else "Unknown",
            "inputs": [{
                "name": input_desc.name,
                "type": str(input_desc.type)
            } for input_desc in spec.description.input],
            "outputs": [{
                "name": output_desc.name,
                "type": str(output_desc.type)
            } for output_desc in spec.description.output]
        }
        
        logger.info(f"Model validated successfully: {model_details['description']}")
        logger.info(f"Inputs: {json.dumps([i['name'] for i in model_details['inputs']])}")
        logger.info(f"Outputs: {json.dumps([o['name'] for o in model_details['outputs']])}")
        
        # Verify the model has expected inputs
        input_names = [i["name"] for i in model_details["inputs"]]
        missing_inputs = [name for name in MODEL_INFO["expected_inputs"] if name not in input_names]
        
        if missing_inputs:
            logger.warning(f"Model is missing expected inputs: {missing_inputs}")
        
        return True, model_details
    
    except ImportError as e:
        logger.error(f"Failed to import coremltools: {str(e)}")
        logger.error("Make sure coremltools is installed correctly")
        return False, {"error": f"Import error: {str(e)}"}
    
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False, {"error": f"Validation error: {str(e)}"}

def verify_file_size(model_path: str) -> bool:
    """
    Verify that the model file has a reasonable size.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        bool: True if file size is reasonable, False otherwise
    """
    try:
        file_size = os.path.getsize(model_path)
        # Most CoreML models are at least a few MB
        min_size = 1024 * 1024  # 1 MB
        
        if file_size < min_size:
            logger.warning(f"Model file is suspiciously small: {file_size} bytes")
            return False
        
        logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
        return True
    
    except Exception as e:
        logger.error(f"Error checking file size: {str(e)}")
        return False

def test_model_prediction(model_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Test the model with a simple prediction to ensure it works.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Tuple[bool, Optional[Dict[str, Any]]]: (success, sample_prediction)
    """
    try:
        import coremltools as ct
        import numpy as np
        
        # Load the model
        model = ct.models.MLModel(model_path)
        
        # Create a sample input
        sample_input = {
            'query_text': 'What is machine learning?',
            'passage_text': 'Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" from data, without being explicitly programmed.'
        }
        
        # Get prediction
        logger.info("Testing model with sample prediction")
        prediction = model.predict(sample_input)
        
        # Convert prediction to JSON-serializable format
        prediction_dict = {}
        for key, value in prediction.items():
            if hasattr(value, 'tolist'):
                prediction_dict[key] = value.tolist()
            else:
                prediction_dict[key] = value
        
        logger.info(f"Sample prediction successful: {json.dumps(prediction_dict, default=str)}")
        return True, prediction_dict
    
    except Exception as e:
        logger.error(f"Error testing model prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def check_model() -> bool:
    """
    Check if the model file exists and is valid.
    
    Returns:
        bool: True if model exists and is valid, False otherwise
    """
    model_path = get_model_path()
    model_dir = os.path.dirname(model_path)
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        logger.error(f"""
=================================================================
ERROR: CoreML model file not found
=================================================================
The model file should be placed in the following location:
{model_path}

This model file is stored using Git LFS and should be included 
in the repository. If you're not seeing the file, make sure:

1. You have Git LFS installed: https://git-lfs.github.com
2. You've pulled the repository with Git LFS enabled:
   git lfs pull

If you are a developer who has the model file separately, 
simply copy it to the path above.
=================================================================
        """)
        return False
    
    logger.info(f"Model file found at {model_path}")
    
    # Verify file size
    if not verify_file_size(model_path):
        logger.warning("Model file has suspicious size")
        return False
    
    # Validate the model
    success, model_details = validate_model(model_path)
    
    if not success:
        logger.error("Model validation failed")
        return False
    
    # Test the model with a prediction
    pred_success, prediction = test_model_prediction(model_path)
    
    if not pred_success:
        logger.error("Model prediction test failed")
        return False
    
    # Save model details
    try:
        details_path = os.path.join(model_dir, "model_details.json")
        with open(details_path, 'w') as f:
            json.dump({
                "model_info": MODEL_INFO,
                "model_details": model_details,
                "sample_prediction": prediction,
                "check_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, default=str)
        
        logger.info("Model details saved to model_details.json")
    except Exception as e:
        logger.warning(f"Failed to save model details: {str(e)}")
    
    logger.info("Model check completed successfully")
    return True

# For backwards compatibility with existing code
def download_model() -> bool:
    """
    Legacy function name kept for compatibility.
    This function no longer downloads the model but checks if it exists locally.
    
    Returns:
        bool: True if model exists and is valid, False otherwise
    """
    return check_model()

if __name__ == "__main__":
    success = check_model()
    if success:
        print(f"Model check successful - CoreML model is valid and ready to use")
    else:
        print(f"Model check failed - Please ensure the CoreML model file is properly installed")
