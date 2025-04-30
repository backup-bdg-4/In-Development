import os
import json
import logging
import traceback
import time
from typing import List, Dict, Any, Optional
import numpy as np
import coremltools as ct
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import requests
import sys
from prometheus_fastapi_instrumentator import Instrumentator

# Import configuration and utilities
from .config import settings
from .utils.security_utils import setup_security, get_client_info, rate_limit_ip_and_tokens
from .utils.code_utils import contains_code, is_code_question, format_code_for_response
from .utils.response_utils import enhance_response, categorize_query, log_query_response
from .utils.search_utils import search_web, SEARCH_CACHE

# Configure logging based on settings
logging_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format=settings.logging.format,
    filename=settings.logging.log_file if settings.logging.log_to_file else None
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title=settings.app_name,
    description="An AI-powered conversational API using Apple's CoreML for natural language processing and question answering",
    version=settings.version,
    debug=settings.debug,
    docs_url="/api/docs" if settings.debug or settings.environment != "production" else None,
    redoc_url="/api/redoc" if settings.debug or settings.environment != "production" else None
)

# Add CORS middleware with origins from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup security features (rate limiting, secure headers, etc.)
setup_security(app)

# Add metrics instrumentation if not in production
if settings.environment != "production":
    Instrumentator().instrument(app).expose(app)

# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} version {settings.version} in {settings.environment} mode")
    
    # Import nltk resources if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        logger.info("NLTK resources loaded")
    except Exception as e:
        logger.warning(f"Failed to load NLTK resources: {e}")
    
    # Check initial model status
    global model, model_loaded
    if model_loaded:
        logger.info("Model pre-loaded successfully")
    else:
        logger.warning("Model not pre-loaded, will be loaded on first request")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")
    
    # Clear any global resources
    if 'model' in globals() and model is not None:
        try:
            model = None
            logger.info("Model released from memory")
        except Exception as e:
            logger.error(f"Error releasing model: {e}")

# Add redirect from root to docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    if settings.debug or settings.environment != "production":
        return RedirectResponse(url="/api/docs")
    else:
        return {"status": "healthy", "service": settings.app_name}

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
    
    # If model doesn't exist, show clear error message
    if not model_status["exists"]:
        error_msg = f"""
=================================================================
ERROR: CoreML model file not found at {MODEL_PATH}
=================================================================
The model file should be placed at the location above.

This model file should be stored using Git LFS in the repository.
If you're not seeing the file, make sure:

1. You have Git LFS installed: https://git-lfs.github.com
2. You've pulled the repository with Git LFS enabled:
   git lfs pull

If you have the model file separately, copy it to the path above.
=================================================================
"""
        logger.error(error_msg)
        model_status["last_error"] = f"Model file not found at {MODEL_PATH}. Please ensure the CoreML model is properly installed."
        return False
    
    # Model file exists, verify it using check_model
    try:
        # Add the parent directory to sys.path to import check_model
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        from download_model import check_model
        
        # Check if the model is valid
        logger.info(f"Verifying model at {MODEL_PATH}...")
        if not check_model():
            error_msg = "Model verification failed. The model file exists but may be corrupted."
            logger.error(error_msg)
            model_status["last_error"] = error_msg
            return False
            
        logger.info("Model verification successful, now loading...")
    
    except ImportError as e:
        error_msg = f"Could not import model verification module: {str(e)}"
        logger.error(error_msg)
        model_status["last_error"] = error_msg
        return False
    
    except Exception as e:
        error_msg = f"Unexpected error during model verification: {str(e)}"
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
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def process_query(request: QueryRequest, request_obj: Request):
    """Process a query using the ML model"""
    global model, model_loaded
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request_obj)
    
    # Validate and sanitize the query
    sanitized_query = sanitize_input(request.query)
    is_valid, error_message = validate_query(sanitized_query)
    
    if not is_valid:
        logger.warning(f"Invalid query from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        logger.error("Model not loaded - cannot process query")
        raise HTTPException(status_code=503, detail="AI model not loaded. Please try again later.")
    
    # Track processing for performance metrics
    start_time = time.time()
    search_results = []
    
    try:
        # Determine if this is a code-related query
        query_category = categorize_query(sanitized_query)
        is_code_question = query_category.get("is_code_question", False)
        
        # Get context - either from request or web search
        context = request.context
        
        # If web search is enabled and no context is provided, fetch context from the web
        if request.web_search and settings.web_search.enabled and (not context or context.strip() == ""):
            search_query = request.search_query or sanitized_query
            logger.info(f"Performing web search for: {search_query}")
            context = await search_web(search_query)
            search_results = SEARCH_CACHE.get(search_query.lower().strip(), {}).get("results", [])
        
        if not context or context.strip() == "":
            context = settings.ai_response.default_context
        
        # Prepare input for the model
        model_input = {
            'query_text': sanitized_query,
            'passage_text': context
        }
        
        # Log the input
        logger.info(f"Model input: query_length={len(sanitized_query)}, context_length={len(context)}")
        
        # Set a timeout for the prediction
        try:
            # Get prediction from the model
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                prediction = model.predict(model_input)
        except asyncio.TimeoutError:
            logger.error(f"Model prediction timed out after {settings.model.predict_timeout_seconds} seconds")
            raise HTTPException(
                status_code=503, 
                detail=f"AI model is taking too long to respond. Please try a simpler query."
            )
        
        # Extract the answer from the prediction
        result = extract_answer(
            prediction, 
            context, 
            query=sanitized_query,
            include_confidence=True
        )
        
        # Determine intent from the query
        intent = determine_intent(sanitized_query)
        
        # Build the response
        response = {
            "answer": result["answer"],
            "intent": intent,
            "confidence": result["confidence"],
            "is_code_question": is_code_question,
            "context_used": True if (context and context != settings.ai_response.default_context) else False,
            "method": result["method_used"],
            "processing_time": time.time() - start_time
        }
        
        # Log the interaction for monitoring
        log_query_response(
            sanitized_query, 
            result["answer"], 
            metadata={
                "client_id": client_info["client_id"],
                "intent": intent,
                "confidence": result["confidence"],
                "method_used": result["method_used"],
                "processing_time": time.time() - start_time,
                "is_code_question": is_code_question,
                "context_used": bool(context and context != settings.ai_response.default_context)
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the failed interaction
        log_query_response(
            sanitized_query, 
            "ERROR", 
            metadata={
                "client_id": client_info["client_id"],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

@app.post("/api/chat/session", response_model=ChatSession)
async def create_chat_session():
    """Create a new chat session"""
    session_id = generate_session_id()
    return {"messages": [], "session_id": session_id}

@app.post("/api/chat/{session_id}", response_model=ChatMessage)
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def chat(session_id: str, message: ChatMessage, request: Request):
    """Add a message to a chat session and get a response"""
    global model, model_loaded
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request)
    logger.info(f"Chat request from client {client_info['client_id']} for session {session_id}")
    
    # Validate and sanitize the message content
    sanitized_content = sanitize_input(message.content)
    is_valid, error_message = validate_query(sanitized_content)
    
    if not is_valid:
        logger.warning(f"Invalid chat message from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Track processing for performance metrics
    start_time = time.time()
    
    # Try to reload the model if it's not loaded
    if model is None and not model_loaded:
        model_loaded = load_model()
    
    if model is None:
        logger.error("Model not loaded - cannot process chat message")
        raise HTTPException(status_code=503, detail="AI model not loaded. Please try again later.")
    
    try:
        # Process the user message
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be sent")
        
        # Analyze the query to determine approach
        query_category = categorize_query(sanitized_content)
        is_code_query = query_category.get("is_code_question", False)
        
        # Determine intent - either from message or by analyzing content
        intent = message.intent if message.intent else determine_intent(sanitized_content)
        
        # Decide whether to use web search
        use_web_search = (
            settings.web_search.enabled and 
            settings.ai_response.enable_web_search_by_default and
            ("search" in intent.lower() or "web_search" in intent.lower())
        )
        
        # Get context from web if needed
        context = ""
        search_results = []
        
        if use_web_search:
            logger.info(f"Performing web search for chat: {sanitized_content[:50]}...")
            context = await search_web(sanitized_content)
            # Get the search results for potential citation
            search_query = sanitized_content.lower().strip()
            search_results = SEARCH_CACHE.get(search_query, {}).get("results", [])
        
        # If no context from search, use default
        if not context or context.strip() == "":
            context = settings.ai_response.default_context
            
            # For code questions, add a more specific context
            if is_code_query:
                lang = query_category.get("language")
                lang_context = f" I can help with {lang} code." if lang else ""
                context = f"I'll help you with your coding question.{lang_context} " + context
        
        # Prepare input for the model
        model_input = {
            'query_text': sanitized_content,
            'passage_text': context
        }
        
        # Log the model input for debugging
        logger.info(f"Chat model input: query_length={len(sanitized_content)}, context_length={len(context)}")
        
        # Set a timeout for the prediction
        try:
            # Get prediction from the model
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                prediction = model.predict(model_input)
        except asyncio.TimeoutError:
            logger.error(f"Model prediction timed out after {settings.model.predict_timeout_seconds} seconds")
            raise HTTPException(
                status_code=503, 
                detail=f"AI model is taking too long to respond. Please try a simpler message."
            )
        
        # Extract the answer from the prediction
        result = extract_answer(
            prediction, 
            context, 
            query=sanitized_content,
            include_confidence=True
        )
        
        # Create assistant response
        response = ChatMessage(
            role="assistant",
            content=result["answer"],
            intent=intent,
            timestamp=get_current_timestamp()
        )
        
        # Log the interaction for monitoring
        log_query_response(
            sanitized_content, 
            result["answer"], 
            metadata={
                "client_id": client_info["client_id"],
                "session_id": session_id,
                "intent": intent,
                "confidence": result["confidence"],
                "method_used": result["method_used"],
                "processing_time": time.time() - start_time,
                "is_code_question": is_code_query,
                "context_used": bool(context and context != settings.ai_response.default_context),
                "web_search_used": use_web_search
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the failed interaction
        log_query_response(
            sanitized_content if 'sanitized_content' in locals() else "ERROR", 
            "ERROR", 
            metadata={
                "client_id": client_info["client_id"],
                "session_id": session_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        
        # Return a user-friendly error message
        error_message = settings.ai_response.fallback_responses.get("error",
            "I encountered an error processing your message. Please try again with a different question.")
            
        return ChatMessage(
            role="assistant",
            content=error_message,
            intent="error",
            timestamp=get_current_timestamp()
        )

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

from .utils.code_utils import contains_code, is_code_question, format_code_for_response
from .utils.response_utils import enhance_response, categorize_query, log_query_response
from .utils.search_utils import search_web
from .utils.security_utils import validate_query, sanitize_input, rate_limit_ip_and_tokens, setup_security, get_client_info
from .config import settings

def extract_answer(prediction: Dict[str, Any], context: str, query: str = "", include_confidence: bool = True) -> Dict[str, Any]:
    """
    Extract the answer from the model prediction and enhance it for the user.
    
    This function handles different output formats from the model:
    1. start_span/end_span indices for extractive QA
    2. Direct answer text fields
    3. Probability/confidence distributions for answer candidates
    
    Args:
        prediction: The model's prediction output
        context: The context text used for the question
        query: The original user query (for better response formatting)
        include_confidence: Whether to include confidence scores
    
    Returns:
        Dict with answer text, confidence level, and metadata
    """
    # Initialize the response structure
    response_data = {
        "answer": "",
        "confidence": 0.0,
        "method_used": "unknown",
        "context_used": bool(context and len(context) > 10),
        "metadata": {}
    }
    
    try:
        # Log the raw prediction type and structure
        logger.info(f"Raw prediction type: {type(prediction)}")
        
        # Create a normalized copy of the prediction for processing
        normalized_pred = {}
        
        # Convert numpy arrays to Python types for better handling
        if hasattr(prediction, 'tolist'):
            # If prediction is itself a numpy array
            prediction = prediction.tolist()
            response_data["metadata"]["prediction_type"] = "numpy_array"
        elif isinstance(prediction, dict):
            # If prediction contains numpy arrays as values
            response_data["metadata"]["prediction_type"] = "dict"
            response_data["metadata"]["prediction_keys"] = list(prediction.keys())
            
            for key, value in prediction.items():
                if hasattr(value, 'tolist'):
                    normalized_pred[key] = value.tolist()
                else:
                    normalized_pred[key] = value
        else:
            response_data["metadata"]["prediction_type"] = str(type(prediction))
        
        # Track if any method succeeded
        extraction_succeeded = False
        raw_answer = ""
        
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
                response_data["metadata"]["start_idx"] = start_idx
                response_data["metadata"]["end_idx"] = end_idx
                
                # Validate indices
                if 0 <= start_idx < len(context) and start_idx <= end_idx < len(context):
                    raw_answer = context[start_idx:end_idx+1].strip()
                    logger.info(f"Successfully extracted answer using span indices: '{raw_answer}'")
                    
                    # Get confidence score if available
                    if 'start_span_probs' in prediction and 'end_span_probs' in prediction:
                        start_prob = max(prediction['start_span_probs'])
                        end_prob = max(prediction['end_span_probs'])
                        confidence = (start_prob + end_prob) / 2
                        response_data["confidence"] = float(confidence)
                    else:
                        response_data["confidence"] = 0.8  # Default high confidence for span-based answers
                    
                    # Return the answer if it's not empty
                    if raw_answer:
                        extraction_succeeded = True
                        response_data["method_used"] = "span_indices"
                else:
                    logger.warning(f"Invalid indices: start={start_idx}, end={end_idx}, context_length={len(context)}")
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error processing start/end spans: {str(e)}")
        
        # APPROACH 2: Look for direct answer fields in the output
        if not extraction_succeeded and isinstance(prediction, dict):
            # Common field names for answers in different models
            answer_field_names = ['answer', 'text', 'response', 'output', 'answer_text', 'prediction']
            
            for field in answer_field_names:
                if field in prediction and isinstance(prediction[field], str) and prediction[field].strip():
                    raw_answer = prediction[field].strip()
                    logger.info(f"Found answer in '{field}' field: '{raw_answer}'")
                    extraction_succeeded = True
                    response_data["method_used"] = f"direct_field_{field}"
                    response_data["confidence"] = 0.9  # Direct fields are usually high confidence
                    break
        
        # APPROACH 3: If prediction is a string itself
        if not extraction_succeeded and isinstance(prediction, str) and prediction.strip():
            raw_answer = prediction.strip()
            logger.info(f"Prediction is a string: '{raw_answer}'")
            extraction_succeeded = True
            response_data["method_used"] = "string_prediction"
            response_data["confidence"] = 0.9
        
        # APPROACH 4: Use maximum probability token from start and end indices
        if not extraction_succeeded and 'start_span_probs' in prediction and 'end_span_probs' in prediction:
            try:
                # Try to find highest probability span
                start_probs = prediction['start_span_probs']
                end_probs = prediction['end_span_probs']
                
                # Get the top 3 most likely start and end positions
                start_indices = sorted(range(len(start_probs)), key=lambda i: -start_probs[i])[:3]
                end_indices = sorted(range(len(end_probs)), key=lambda i: -end_probs[i])[:3]
                
                best_score = 0
                best_answer = ""
                
                # Try different combinations
                for start_idx in start_indices:
                    for end_idx in end_indices:
                        if start_idx <= end_idx and end_idx < len(context):
                            answer = context[start_idx:end_idx+1].strip()
                            if answer and len(answer) > 2:  # Minimum answer length
                                score = (start_probs[start_idx] + end_probs[end_idx]) / 2
                                if score > best_score:
                                    best_score = score
                                    best_answer = answer
                
                if best_answer:
                    raw_answer = best_answer
                    logger.info(f"Extracted answer from probabilities: '{raw_answer}'")
                    extraction_succeeded = True
                    response_data["method_used"] = "probability_span"
                    response_data["confidence"] = float(best_score)
            except Exception as e:
                logger.warning(f"Error extracting answer from probabilities: {str(e)}")
        
        # FALLBACK: Generate a response based on the context
        if not extraction_succeeded and context and len(context) > 20:
            # Try to provide a meaningful response from the context
            context_start = context[:settings.ai_response.max_response_length//2].strip()
            if len(context) > settings.ai_response.max_response_length//2:
                context_start += "..."
                
            logger.info(f"Using context beginning as fallback")
            raw_answer = f"Based on the available information: {context_start}"
            extraction_succeeded = True
            response_data["method_used"] = "context_fallback"
            response_data["confidence"] = 0.3  # Low confidence for fallbacks
        
        # Final fallback if all else fails
        if not extraction_succeeded or not raw_answer:
            logger.warning("Could not extract a good answer from model prediction")
            
            # Check if it's a code question to provide a more specific fallback
            if query and is_code_question(query):
                raw_answer = settings.ai_response.fallback_responses.get("coding", 
                    "I'm not able to generate the code for this specific request. Could you provide more details?")
            else:
                raw_answer = settings.ai_response.fallback_responses.get("default",
                    "I'm unable to provide a specific answer based on the available information. Please try rephrasing your question.")
            
            response_data["method_used"] = "final_fallback"
            response_data["confidence"] = 0.1  # Very low confidence
        
        # Set the extracted answer
        response_data["answer"] = raw_answer
        
        # Enhance the response if we have the query
        if query:
            enhanced_answer = enhance_response(
                raw_answer, 
                query, 
                intent=determine_intent(query),
                confidence=response_data["confidence"] if include_confidence else None
            )
            response_data["answer"] = enhanced_answer
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log detailed error diagnostics
        logger.error(f"Prediction type: {type(prediction)}")
        logger.error(f"Context length: {len(context) if context else 0}")
        if isinstance(prediction, dict):
            logger.error(f"Prediction keys: {list(prediction.keys())}")
        
        # Return error response
        error_message = settings.ai_response.fallback_responses.get("error",
            "I encountered an error while processing your question. Please try again with a different phrasing.")
        
        response_data["answer"] = error_message
        response_data["method_used"] = "error_handler"
        response_data["confidence"] = 0.0
        
        return response_data

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
