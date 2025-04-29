import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
import numpy as np
import coremltools as ct
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import requests
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Backdoor AI - ML Model API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine model path from environment variable or default location
model_data_path = os.environ.get('MODEL_DATA_PATH', None)
if model_data_path:
    MODEL_PATH = os.path.join(model_data_path, "BERTSQUADFP16.mlmodel")
else:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "BERTSQUADFP16.mlmodel")

# Global variable to store the loaded model
model = None

# Model status tracking
model_status = {
    "loaded": False,
    "path": MODEL_PATH,
    "exists": False,
    "last_error": None,
    "load_attempts": 0,
    "last_attempt_time": None,
    "details": {}
}

def load_model(force_reload=False):
    """
    Load the CoreML model for inference.
    
    Args:
        force_reload (bool): If True, reload the model even if it's already loaded
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model, model_status
    
    # Track attempt
    model_status["load_attempts"] += 1
    model_status["last_attempt_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # If model is already loaded and no force reload, return True
    if model is not None and not force_reload:
        logger.info("Model already loaded, skipping load")
        return True
    
    # Check if model file exists
    model_status["exists"] = os.path.exists(MODEL_PATH)
    
    # If model doesn't exist, try to download it
    if not model_status["exists"]:
        logger.warning(f"Model not found at {MODEL_PATH}. Attempting to download...")
        
        # Add the parent directory to sys.path to import download_model
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        try:
            # Import the download_model function
            from download_model import download_model
            
            # Try to download the model
            logger.info("Starting model download...")
            if not download_model():
                error_msg = "Failed to download model"
                logger.error(error_msg)
                model_status["last_error"] = error_msg
                return False
            
            # Update model existence status after download
            model_status["exists"] = os.path.exists(MODEL_PATH)
            if not model_status["exists"]:
                error_msg = "Model download reported success but file doesn't exist"
                logger.error(error_msg)
                model_status["last_error"] = error_msg
                return False
                
            logger.info("Model downloaded successfully, now loading...")
        
        except ImportError as e:
            error_msg = f"Could not import download_model module: {str(e)}"
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            return False
        
        except Exception as e:
            error_msg = f"Unexpected error during model download: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            model_status["last_error"] = error_msg
            return False
    
    # Load the ML model
    try:
        # Clear any previous model from memory
        if model is not None:
            model = None
        
        logger.info(f"Loading model from {MODEL_PATH}")
        
        # Try importing coremltools if not already imported
        try:
            import coremltools as ct
        except ImportError as e:
            error_msg = f"Failed to import coremltools: {str(e)}"
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            return False
        
        # Load the model
        start_time = time.time()
        model = ct.models.MLModel(MODEL_PATH)
        load_time = time.time() - start_time
        
        # Get model details
        spec = model.get_spec()
        
        # Store model metadata
        model_status["details"] = {
            "description": spec.description.metadata.shortDescription if hasattr(spec.description.metadata, "shortDescription") else "Unknown",
            "author": spec.description.metadata.author if hasattr(spec.description.metadata, "author") else "Unknown",
            "load_time_sec": load_time,
            "inputs": [input_desc.name for input_desc in spec.description.input],
            "outputs": [output_desc.name for output_desc in spec.description.output]
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model description: {model_status['details']['description']}")
        logger.info(f"Model inputs: {model_status['details']['inputs']}")
        logger.info(f"Model outputs: {model_status['details']['outputs']}")
        
        # Test model with a simple prediction
        test_success = _test_model_basic_prediction()
        if not test_success:
            error_msg = "Model loaded but failed basic prediction test"
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            model_status["loaded"] = False
            return False
        
        # Update status
        model_status["loaded"] = True
        model_status["last_error"] = None
        return True
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_status["last_error"] = error_msg
        model_status["loaded"] = False
        return False

def _test_model_basic_prediction():
    """
    Test the model with a basic prediction to ensure it's working correctly.
    
    Returns:
        bool: True if prediction succeeds, False otherwise
    """
    global model
    
    if model is None:
        logger.error("Cannot test model - model not loaded")
        return False
    
    try:
        # Create a simple test input
        test_input = {
            'query_text': 'What is AI?',
            'passage_text': 'Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.'
        }
        
        # Try a prediction
        logger.info("Testing model with basic prediction")
        prediction = model.predict(test_input)
        
        # Check if prediction has expected format
        if isinstance(prediction, dict) and prediction:
            logger.info("Basic model prediction successful")
            return True
        else:
            logger.error(f"Unexpected prediction format: {type(prediction)}")
            return False
            
    except Exception as e:
        logger.error(f"Error during model test prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load the model on startup
model_loaded = load_model()

# Schedule periodic model checks
def schedule_model_checks():
    """Schedule periodic checks to ensure model is loaded"""
    import threading
    
    def check_model():
        global model, model_loaded, model_status
        
        # If model is not loaded, try to reload it
        if model is None or not model_status["loaded"]:
            logger.info("Scheduled check: Attempting to reload model")
            model_loaded = load_model()
        
        # Schedule the next check
        check_timer = threading.Timer(300, check_model)  # Check every 5 minutes
        check_timer.daemon = True
        check_timer.start()
    
    # Start the first check
    initial_timer = threading.Timer(60, check_model)  # First check after 1 minute
    initial_timer.daemon = True
    initial_timer.start()

# Start the scheduled checks
schedule_model_checks()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    web_search: bool = False
    search_query: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    intent: Optional[str] = None
    timestamp: Optional[str] = None

class ChatSession(BaseModel):
    messages: List[ChatMessage]
    session_id: str

@app.get("/")
async def root():
    """
    Health check endpoint with detailed model status information.
    This provides diagnostics that help the frontend understand
    the status of the model and what might be wrong.
    """
    global model, model_loaded, model_status
    
    # Try to reload the model if it's not loaded
    if model is None and not model_status["loaded"] and not model_loaded:
        logger.info("Model not loaded, attempting to load in health check")
        model_loaded = load_model()
    
    # Prepare status response
    response = {
        "status": "healthy" if model_status["loaded"] else "degraded",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "loaded": model_status["loaded"],
            "file_exists": os.path.exists(MODEL_PATH),
            "path": MODEL_PATH,
            "load_attempts": model_status["load_attempts"],
            "last_attempt": model_status["last_attempt_time"],
        },
        "api_version": "1.0.0"
    }
    
    # If we have an error, include it
    if model_status["last_error"]:
        response["model"]["error"] = model_status["last_error"]
    
    # Include model details if available
    if model_status["details"]:
        response["model"]["details"] = model_status["details"]
    
    return response

@app.post("/api/query", response_model=Dict[str, Any])
async def process_query(request: QueryRequest):
    """Process a query using the ML model"""
    global model, model_loaded
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # If web search is enabled and no context is provided, fetch context from the web
        context = request.context
        if request.web_search and (not context or context.strip() == ""):
            search_query = request.search_query or request.query
            context = await search_web(search_query)
        
        if not context:
            context = "No context provided. I'll try to answer based on my knowledge."
        
        # Prepare input for the model
        model_input = {
            'query_text': request.query,
            'passage_text': context
        }
        
        # Get prediction from the model
        prediction = model.predict(model_input)
        
        # Extract the answer from the prediction
        answer = extract_answer(prediction, context)
        
        # Determine intent from the query
        intent = determine_intent(request.query)
        
        return {
            "answer": answer,
            "intent": intent,
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/chat/session", response_model=ChatSession)
async def create_chat_session():
    """Create a new chat session"""
    session_id = generate_session_id()
    return {"messages": [], "session_id": session_id}

@app.post("/api/chat/{session_id}", response_model=ChatMessage)
async def chat(session_id: str, message: ChatMessage):
    """Add a message to a chat session and get a response"""
    global model, model_loaded
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process the user message
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be sent")
        
        # Determine intent
        intent = determine_intent(message.content)
        
        # Get context from web if needed
        context = await search_web(message.content) if "search" in intent.lower() else ""
        
        # Prepare input for the model
        model_input = {
            'query_text': message.content,
            'passage_text': context or "Please provide a helpful response based on your knowledge."
        }
        
        # Log the model input for debugging
        logger.info(f"Model input: {json.dumps(model_input)}")
        
        # Get prediction from the model
        try:
            prediction = model.predict(model_input)
            logger.info(f"Raw prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
        
        # Extract the answer from the prediction
        answer = extract_answer(prediction, context)
        
        # Create assistant response
        response = ChatMessage(
            role="assistant",
            content=answer,
            intent=intent,
            timestamp=get_current_timestamp()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/api/chat/{session_id}/export", response_model=Dict[str, Any])
async def export_chat_session(session_id: str):
    """Export a chat session as JSON"""
    # In a real implementation, you would retrieve the session from a database
    # For this example, we'll return a sample session
    sample_session = {
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how can you help me?",
                "intent": "greeting",
                "timestamp": "2023-11-01T12:00:00Z"
            },
            {
                "role": "assistant",
                "content": "Hi there! I'm Backdoor AI, and I'm here to help answer your questions and provide information. Feel free to ask me anything!",
                "intent": "greeting_response",
                "timestamp": "2023-11-01T12:00:05Z"
            }
        ]
    }
    
    return sample_session

# Helper functions
async def search_web(query: str) -> str:
    """Search the web for information related to the query"""
    try:
        async with aiohttp.ClientSession() as session:
            # Use a search engine API or scrape search results
            # For this example, we'll use a simple approach with DuckDuckGo
            search_url = f"https://duckduckgo.com/html/?q={query}"
            async with session.get(search_url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results
                    results = []
                    for result in soup.select(".result__body"):
                        title_elem = result.select_one(".result__title")
                        snippet_elem = result.select_one(".result__snippet")
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True)
                            results.append(f"{title}: {snippet}")
                    
                    return "\n\n".join(results) if results else "No relevant information found."
                else:
                    return "Unable to search the web at this time."
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return "Error occurred while searching the web."

def extract_answer(prediction: Dict[str, Any], context: str) -> str:
    """
    Extract the answer from the model prediction.
    
    This function handles different output formats from the model:
    1. start_span/end_span indices for extractive QA
    2. Direct answer text fields
    3. Probability/confidence distributions for answer candidates
    
    Args:
        prediction: The model's prediction output
        context: The context text used for the question
    
    Returns:
        str: The extracted answer or appropriate fallback message
    """
    try:
        # Log the raw prediction type and structure
        logger.info(f"Raw prediction type: {type(prediction)}")
        
        # Create a normalized copy of the prediction for processing
        normalized_pred = {}
        
        # Convert numpy arrays to Python types for better handling
        if hasattr(prediction, 'tolist'):
            # If prediction is itself a numpy array
            prediction = prediction.tolist()
        elif isinstance(prediction, dict):
            # If prediction contains numpy arrays as values
            for key, value in prediction.items():
                if hasattr(value, 'tolist'):
                    normalized_pred[key] = value.tolist()
                else:
                    normalized_pred[key] = value
        
        # Log the normalized prediction
        logger.info(f"Normalized prediction: {json.dumps(normalized_pred, default=str)}")
        
        # APPROACH 1: Handle extractive QA output format (start_span/end_span)
        # This is the expected format for many BERT-based QA models
        if isinstance(prediction, dict) and 'start_span' in prediction and 'end_span' in prediction:
            try:
                # Handle array or single value
                start_idx = prediction['start_span'][0] if isinstance(prediction['start_span'], (list, tuple)) else prediction['start_span']
                end_idx = prediction['end_span'][0] if isinstance(prediction['end_span'], (list, tuple)) else prediction['end_span']
                
                # Convert to integers if they're numpy values or floats
                start_idx = int(float(start_idx))
                end_idx = int(float(end_idx))
                
                logger.info(f"Extracted indices: start={start_idx}, end={end_idx}, context_length={len(context)}")
                
                # Validate indices
                if 0 <= start_idx < len(context) and start_idx <= end_idx < len(context):
                    answer = context[start_idx:end_idx+1].strip()
                    logger.info(f"Successfully extracted answer using span indices: '{answer}'")
                    
                    # Return the answer if it's not empty
                    if answer:
                        return answer
                else:
                    logger.warning(f"Invalid indices: start={start_idx}, end={end_idx}, context_length={len(context)}")
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error processing start/end spans: {str(e)}")
        
        # APPROACH 2: Look for direct answer fields in the output
        if isinstance(prediction, dict):
            # Common field names for answers in different models
            answer_field_names = ['answer', 'text', 'response', 'output', 'answer_text', 'prediction']
            
            for field in answer_field_names:
                if field in prediction and isinstance(prediction[field], str) and prediction[field].strip():
                    logger.info(f"Found answer in '{field}' field: '{prediction[field]}'")
                    return prediction[field].strip()
        
        # APPROACH 3: If we have a logits/probabilities output, find the most likely answer
        if isinstance(prediction, dict) and ('logits' in prediction or 'probabilities' in prediction or 'scores' in prediction):
            # For models that return probability distributions
            # This would need more specific implementation based on the model's output format
            logger.info("Model returned probability distribution, but specific handling is not implemented")
        
        # APPROACH 4: If prediction is a string itself (rare but possible)
        if isinstance(prediction, str) and prediction.strip():
            logger.info(f"Prediction is a string: '{prediction}'")
            return prediction.strip()
        
        # APPROACH 5: Use maximum probability token from start and end indices
        if 'start_span_probs' in prediction and 'end_span_probs' in prediction:
            try:
                # Try to find highest probability span
                start_probs = prediction['start_span_probs']
                end_probs = prediction['end_span_probs']
                
                # Get the top 3 most likely start and end positions
                start_indices = sorted(range(len(start_probs)), key=lambda i: -start_probs[i])[:3]
                end_indices = sorted(range(len(end_probs)), key=lambda i: -end_probs[i])[:3]
                
                # Try different combinations
                for start_idx in start_indices:
                    for end_idx in end_indices:
                        if start_idx <= end_idx and end_idx < len(context):
                            answer = context[start_idx:end_idx+1].strip()
                            if answer and len(answer) > 2:  # Minimum answer length
                                logger.info(f"Extracted answer from probabilities: '{answer}'")
                                return answer
            except Exception as e:
                logger.warning(f"Error extracting answer from probabilities: {str(e)}")
        
        # FALLBACK: Generate a response based on the context
        if context and len(context) > 20:
            # Try to provide a meaningful response from the context
            # Use the first 100-150 characters as a generic response
            context_start = context[:150].strip()
            if len(context) > 150:
                context_start += "..."
                
            logger.info(f"Using context beginning as fallback: '{context_start}'")
            return f"Based on the information provided: {context_start}"
            
        # Final fallback
        logger.warning("Could not extract a good answer from model prediction")
        return "I'm unable to provide a specific answer based on the available information. Please try rephrasing your question."
    
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log detailed error diagnostics
        logger.error(f"Prediction type: {type(prediction)}")
        logger.error(f"Context length: {len(context)}")
        if isinstance(prediction, dict):
            logger.error(f"Prediction keys: {list(prediction.keys())}")
        
        return "I encountered an error while processing your question. Please try again with a different phrasing."

def determine_intent(query: str) -> str:
    """Determine the intent of the user's query"""
    query = query.lower()
    
    if any(word in query for word in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    elif any(word in query for word in ["search", "find", "look up", "google"]):
        return "web_search"
    elif any(word in query for word in ["what is", "who is", "explain", "define"]):
        return "definition"
    elif any(word in query for word in ["how to", "how do i"]):
        return "instruction"
    elif any(word in query for word in ["why", "reason"]):
        return "explanation"
    else:
        return "general_query"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

def get_current_timestamp() -> str:
    """Get the current timestamp in ISO format"""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
