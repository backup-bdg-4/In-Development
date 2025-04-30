"""
Remote Model Server for BERTSQUADFP16.mlmodel

This is a standalone FastAPI application that loads the CoreML model
and provides prediction endpoints. It should be deployed separately
from the main application.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CoreML Model Server",
    description="A remote server for CoreML model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API key for authentication
API_KEY = os.environ.get("MODEL_SERVER_API_KEY", "your-api-key-here")

# Global model variable
model = None
model_path = os.environ.get("MODEL_PATH", "BERTSQUADFP16.mlmodel")
model_loaded = False
model_load_time = None

# Input and output models
class PredictionRequest(BaseModel):
    query_text: str
    passage_text: str

class PredictionResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    start_index: int = 0
    end_index: int = 0
    error: Optional[str] = None

# Authentication dependency
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY or API_KEY == "your-api-key-here":
        # No API key set, skip authentication
        return True
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )
    
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True

# Load the model
def load_model():
    global model, model_loaded, model_load_time
    
    if model_loaded:
        logger.info("Model already loaded")
        return True
    
    try:
        logger.info(f"Loading model from {model_path}...")
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Import coremltools here to avoid loading it if not needed
        import coremltools as ct
        
        # Load the model
        model = ct.models.MLModel(model_path)
        load_time = time.time() - start_time
        model_load_time = load_time
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting CoreML Model Server")
    
    # Load the model in a background thread to avoid blocking startup
    import threading
    
    def load_model_thread():
        success = load_model()
        if success:
            logger.info("Model loaded successfully in background thread")
        else:
            logger.error("Failed to load model in background thread")
    
    thread = threading.Thread(target=load_model_thread)
    thread.daemon = True
    thread.start()
    logger.info("Started background thread for model loading")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "model_load_time_sec": model_load_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Model status endpoint
@app.get("/model/status")
async def model_status():
    return {
        "model_loaded": model_loaded,
        "model_path": model_path,
        "model_load_time_sec": model_load_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    global model, model_loaded
    
    if not model_loaded:
        success = load_model()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
    
    try:
        # Prepare input for the model
        model_input = {
            'query_text': request.query_text,
            'passage_text': request.passage_text
        }
        
        # Make prediction
        prediction = model.predict(model_input)
        
        # Extract start and end indices
        start_idx = int(prediction.get('start_index', 0))
        end_idx = int(prediction.get('end_index', 0))
        
        # Extract the answer from the context
        answer = request.passage_text[start_idx:end_idx+1] if start_idx <= end_idx else ''
        
        # Calculate confidence score
        confidence = float(prediction.get('confidence', 0.0))
        
        # Return the response
        return {
            'answer': answer,
            'confidence': confidence,
            'start_index': start_idx,
            'end_index': end_idx
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

