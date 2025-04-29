import os
import requests
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

# Dropbox direct download link for the .mlmodel file
# Note: This should be a direct download link from Dropbox (usually ends with ?dl=1)
DROPBOX_LINK = "https://www.dropbox.com/scl/fi/w4iclrvil6vh39mg6j7pl/BERTSQUADFP16.mlmodel?rlkey=vbrr9jjvsam1xg9i4i19pkdra&st=ho9dyrm6&dl=1"

# Alternate download URLs in case the primary one fails
ALTERNATE_LINKS = [
    # Add alternative links here if available
]

# Model information
MODEL_INFO = {
    "name": "BERTSQUADFP16",
    "type": "BERT-based question answering model",
    "expected_inputs": ["query_text", "passage_text"],
    "expected_outputs": ["start_span", "end_span"]
}

def verify_content_type(response: requests.Response) -> bool:
    """
    Verify that the response has the correct content type.
    
    Args:
        response (requests.Response): Response from the download request
        
    Returns:
        bool: True if content type is valid, False otherwise
    """
    content_type = response.headers.get('content-type', '').lower()
    valid_types = ['application/octet-stream', 'application/binary', 'application/x-binary']
    
    # Check if content type is one of the valid types
    if any(valid_type in content_type for valid_type in valid_types):
        return True
    
    # Check if content type looks like a binary file
    if 'binary' in content_type or 'octet' in content_type:
        return True
    
    # Check for HTML content which would indicate an error page
    if 'text/html' in content_type:
        first_chunk = next(response.iter_content(1024), b'')
        if b'<html' in first_chunk.lower():
            logger.error("Response contains HTML instead of binary data")
            logger.debug(f"HTML content: {first_chunk[:200]}")
            return False
    
    logger.warning(f"Unexpected content type: {content_type}")
    # Continue anyway as some servers might return incorrect content type
    return True

def validate_model(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the downloaded model is a valid CoreML model.
    
    Args:
        model_path (str): Path to the downloaded model file
        
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
            "description": spec.description.metadata.shortDescription,
            "author": spec.description.metadata.author,
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
        
        # Verify the model has expected outputs
        output_names = [o["name"] for o in model_details["outputs"]]
        
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
    Verify that the downloaded file has a reasonable size for a model.
    
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

def download_model(max_retries: int = 3, retry_delay: int = 5) -> bool:
    """
    Download the ML model from Dropbox and save it to the appropriate location.
    
    Args:
        max_retries (int): Maximum number of download attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        bool: True if download and validation was successful, False otherwise
    """
    # Define the model directory and file path
    model_dir = os.path.join(os.path.dirname(__file__), "app", "model")
    model_path = os.path.join(model_dir, "BERTSQUADFP16.mlmodel")
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if the model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        
        # Verify file size
        if not verify_file_size(model_path):
            logger.warning("Existing model file has suspicious size. Will attempt to re-download.")
        else:
            # Validate the existing model
            success, model_details = validate_model(model_path)
            
            if success:
                # Test the model with a prediction
                pred_success, _ = test_model_prediction(model_path)
                if pred_success:
                    return True
                else:
                    logger.warning("Existing model failed prediction test. Will attempt to re-download.")
            else:
                logger.warning("Existing model is invalid. Will attempt to re-download.")
    
    # Prepare list of URLs to try
    urls_to_try = [DROPBOX_LINK] + ALTERNATE_LINKS
    
    for url_index, url in enumerate(urls_to_try):
        logger.info(f"Trying download URL {url_index+1}/{len(urls_to_try)}: {url}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt+1}/{max_retries}")
                
                # Send a GET request to the download link
                response = requests.get(url, stream=True, timeout=30)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Verify content type
                    if not verify_content_type(response):
                        logger.warning(f"Invalid content type. Skipping to next URL.")
                        break
                    
                    # Get the total file size
                    total_size = int(response.headers.get('content-length', 0))
                    logger.info(f"Total model size: {total_size / (1024*1024):.2f} MB")
                    
                    # Create a temporary file to download to
                    temp_path = model_path + ".download"
                    
                    # Write the file in chunks
                    with open(temp_path, 'wb') as f:
                        downloaded = 0
                        chunk_size = 1024 * 1024  # 1MB chunks
                        start_time = time.time()
                        
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:  # filter out keep-alive chunks
                                continue
                                
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Calculate and print progress
                            elapsed = time.time() - start_time
                            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                            speed = downloaded / (elapsed * 1024 * 1024) if elapsed > 0 else 0
                            
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) at {speed:.2f} MB/s")
                    
                    logger.info(f"Download completed. Validating model...")
                    
                    # Verify the downloaded file size
                    if not verify_file_size(temp_path):
                        logger.error("Downloaded file has suspicious size. Retrying...")
                        continue
                    
                    # Move the temp file to the final location
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    os.rename(temp_path, model_path)
                    
                    # Validate the downloaded model
                    success, model_details = validate_model(model_path)
                    
                    if success:
                        # Test the model with a prediction
                        pred_success, prediction = test_model_prediction(model_path)
                        
                        if pred_success:
                            logger.info("Model downloaded, validated, and tested successfully")
                            
                            # Save model details for reference
                            details_path = os.path.join(model_dir, "model_details.json")
                            with open(details_path, 'w') as f:
                                json.dump({
                                    "model_info": MODEL_INFO,
                                    "model_details": model_details,
                                    "sample_prediction": prediction,
                                    "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
                                }, f, indent=2, default=str)
                            
                            return True
                        else:
                            logger.error("Model validation succeeded but prediction test failed")
                    else:
                        logger.error(f"Model validation failed: {model_details.get('error', 'Unknown error')}")
                
                else:
                    logger.error(f"Failed to download model. Status code: {response.status_code}")
                    if response.status_code in [403, 404, 410]:
                        logger.error("URL appears to be invalid or expired. Trying next URL.")
                        break
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during download: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                logger.error(traceback.format_exc())
            
            # If we get here, the download failed and we should retry
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying download in {wait_time} seconds...")
                time.sleep(wait_time)
    
    logger.error("All download attempts failed")
    return False

if __name__ == "__main__":
    success = download_model()
    print(f"Model download {'successful' if success else 'failed'}")
