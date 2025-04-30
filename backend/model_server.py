"""
Standalone model server for serving ML models.
This is a lightweight FastAPI server that can be run separately from the main application.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Model Server",
    description="Lightweight model server for ML inference",
    version="1.0.0"
)

# Define request and response models
class PredictionRequest(BaseModel):
    query_text: str
    passage_text: str

class PredictionResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    start_index: int = 0
    end_index: int = 0

# Global model variable
model = None

# Mock prediction function (replace with actual model loading and prediction)
def predict(query: str, context: str) -> Dict[str, Any]:
    """
    Mock prediction function.
    In a real implementation, this would load and use an actual ML model.
    
    Args:
        query: Query text
        context: Context text
        
    Returns:
        Dictionary with prediction result
    """
    # Simple mock implementation that finds the first sentence containing a keyword from the query
    query_words = set(query.lower().split())
    
    # Find the best sentence in the context
    sentences = context.split('.')
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Count matching words
        sentence_words = set(sentence.lower().split())
        matches = len(query_words.intersection(sentence_words))
        
        if matches > best_score:
            best_score = matches
            best_sentence = sentence
    
    # If no good match, use the first sentence
    if not best_sentence and sentences:
        best_sentence = sentences[0].strip()
    
    # Calculate start and end indices
    start_index = context.find(best_sentence) if best_sentence else 0
    end_index = start_index + len(best_sentence) - 1 if best_sentence else 0
    
    # Calculate confidence based on word matches
    confidence = min(0.9, (best_score / max(1, len(query_words))) * 0.9) if best_score > 0 else 0.5
    
    return {
        "answer": best_sentence,
        "confidence": confidence,
        "start_index": start_index,
        "end_index": end_index
    }

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

@app.post("/predict", response_model=PredictionResponse)
async def prediction_endpoint(request: PredictionRequest):
    """
    Prediction endpoint.
    
    Args:
        request: PredictionRequest with query_text and passage_text
        
    Returns:
        PredictionResponse with answer and metadata
    """
    try:
        # Make prediction
        result = predict(request.query_text, request.passage_text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/info")
async def server_info():
    """Server information endpoint."""
    return {
        "name": "Model Server",
        "version": "1.0.0",
        "description": "Lightweight model server for ML inference",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/predict", "method": "POST", "description": "Prediction endpoint"},
            {"path": "/info", "method": "GET", "description": "Server information endpoint"}
        ]
    }

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("MODEL_SERVER_PORT", 5000))
    host = os.environ.get("MODEL_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting model server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

