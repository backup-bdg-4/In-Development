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
import gc
from prometheus_fastapi_instrumentator import Instrumentator

# Import configuration and utilities
from .config import settings
from .utils.security_utils import setup_security, get_client_info, rate_limit_ip_and_tokens
from .utils.code_utils import contains_code, is_code_question, format_code_for_response
from .utils.response_utils import enhance_response, categorize_query, log_query_response
from .utils.search_utils import search_web, SEARCH_CACHE

# Import Jupyter model server (for memory-efficient model loading)
try:
    from .utils.jupyter_model_server import (
        initialize_model_server, 
        predict_with_jupyter, 
        predict_with_jupyter_server,
        shutdown_jupyter_server,
        is_jupyter_server_running
    )
    JUPYTER_MODEL_SERVER_AVAILABLE = True
except ImportError:
    JUPYTER_MODEL_SERVER_AVAILABLE = False
    logging.warning("Jupyter model server not available, falling back to standard model loading")

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
    
    # Check if we're in memory-saving mode
    minimize_memory = os.environ.get('MINIMIZE_MEMORY_USAGE') == 'true'
    running_on_render = os.environ.get('RUNNING_ON_RENDER') == 'true'
    memory_saving_mode = minimize_memory or running_on_render
    
    if memory_saving_mode:
        logger.info("Memory-saving mode active - optimizing startup process")
    
    # Import nltk resources only if not in memory-saving mode
    if not memory_saving_mode:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            logger.info("NLTK resources loaded")
        except Exception as e:
            logger.warning(f"Failed to load NLTK resources: {e}")
    else:
        logger.info("Memory-saving mode: Skipping NLTK resources loading")
    
    # Initialize the application with enhanced setup
    try:
        from .utils.initialization import initialize_app
        init_result = initialize_app()
        
        logger.info(f"Application initialized in {init_result['initialization_time_sec']:.2f} seconds")
        logger.info(f"Environment: {init_result['environment']['environment']}")
        
        if init_result['model']['found']:
            logger.info(f"✅ Model found at {init_result['model']['path']} ({init_result['model']['size_mb']:.2f} MB)")
            
            # Save model path for later use
            app.state.model_path = init_result['model']['path']
        else:
            logger.error(f"❌ Model not found during initialization")
            
        # Save initialization info for health check
        app.state.init_result = init_result
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}")
        logger.error(traceback.format_exc())
        app.state.init_result = {
            "success": False,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Check if we should use Jupyter model server
    use_jupyter_env = os.environ.get('USE_JUPYTER_MODEL_SERVER') == 'true'
    use_jupyter = (memory_saving_mode or use_jupyter_env) and JUPYTER_MODEL_SERVER_AVAILABLE
    app.state.use_jupyter_server = use_jupyter
    
    if use_jupyter_env and not JUPYTER_MODEL_SERVER_AVAILABLE:
        logger.warning("USE_JUPYTER_MODEL_SERVER is set but Jupyter is not available. Falling back to standard model loading.")
    
    if use_jupyter:
        logger.info("Using Jupyter model server for memory-efficient model loading")
        
        # Initialize Jupyter model server in a background thread
        import threading
        
        def init_jupyter_server_thread():
            try:
                model_path = getattr(app.state, 'model_path', '/tmp/model/BERTSQUADFP16.mlmodel')
                logger.info(f"Initializing Jupyter model server with model at {model_path}")
                success = initialize_model_server(model_path)
                if success:
                    logger.info("Jupyter model server initialized successfully")
                    app.state.jupyter_server_ready = True
                else:
                    logger.error("Failed to initialize Jupyter model server")
                    app.state.jupyter_server_ready = False
            except Exception as e:
                logger.error(f"Error initializing Jupyter model server: {str(e)}")
                app.state.jupyter_server_ready = False
        
        # Start initialization in a background thread
        jupyter_thread = threading.Thread(target=init_jupyter_server_thread)
        jupyter_thread.daemon = True
        jupyter_thread.start()
        logger.info("Started background thread for Jupyter model server initialization")
        
        # Force garbage collection to free memory
        try:
            gc.collect()
            logger.info("Memory-saving mode: Garbage collection performed")
        except Exception as e:
            logger.warning(f"Failed to perform garbage collection: {e}")
    else:
        # Force garbage collection to free memory
        try:
            gc.collect()
            logger.info("Memory-saving mode: Garbage collection performed")
        except Exception as e:
            logger.warning(f"Failed to perform garbage collection: {e}")
            
        logger.warning("Standard model loading is disabled. Please use Jupyter model server.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")
    
    # Check if we're using Jupyter model server
    if getattr(app.state, 'use_jupyter_server', False):
        try:
            logger.info("Shutting down Jupyter model server")
            shutdown_jupyter_server()
            logger.info("Jupyter model server shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Jupyter model server: {str(e)}")
    else:
        # Clear any global resources
        logger.info("No Jupyter server to shut down")
    
    # Force garbage collection to free memory
    try:
        gc.collect()
        logger.info("Garbage collection performed during shutdown")
    except Exception as e:
        logger.warning(f"Failed to perform garbage collection: {e}")

# Add redirect from root to docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    if settings.debug or settings.environment != "production":
        return RedirectResponse(url="/api/docs")
    else:
        return {"status": "healthy", "service": settings.app_name}

# Import model utilities
from .utils.model_utils import find_model_file, validate_model, ensure_model_availability

# Model status tracking
model_status = {
    "loaded": True,
    "path": os.environ.get('MODEL_DATA_PATH', '/tmp/model'),
    "exists": True,
    "last_error": None,
    "load_attempts": 0,
    "last_attempt_time": None,
    "details": {},
    "alternate_paths": [],
    "search_paths_checked": []
}

def load_model(force_reload=False):
    """
    Placeholder function for compatibility.
    With Jupyter model server, we do not need to load the model in the main application.
    
    Args:
        force_reload (bool): Not used
        
    Returns:
        bool: Always returns True
    """
    global model_status
    
    # Track attempt for compatibility
    model_status["load_attempts"] += 1
    model_status["last_attempt_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Update status
    model_status["exists"] = True
    model_status["path"] = os.environ.get("MODEL_DATA_PATH", "/tmp/model")
    model_status["memory_saving_mode"] = True
    model_status["loaded"] = True
    model_status["load_time"] = 0
    model_status["last_loaded"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("Using Jupyter model server - model loading handled by server")
    return True

# Initialize model status
load_model()

# Schedule periodic Jupyter server checks
def schedule_model_checks():
    """Schedule periodic checks to ensure Jupyter model server is running"""
    import threading
    
    def check_jupyter_server():
        global model_status
        
        # Check if Jupyter server is running
        if JUPYTER_MODEL_SERVER_AVAILABLE:
            try:
                server_running = is_jupyter_server_running()
                if not server_running:
                    logger.warning("Jupyter model server not running, attempting to initialize")
                    initialize_model_server()
            except Exception as e:
                logger.error(f"Error checking Jupyter server status: {str(e)}")
        
        # Schedule the next check
        check_timer = threading.Timer(300, check_jupyter_server)  # Check every 5 minutes
        check_timer.daemon = True
        check_timer.start()
    
    # Start the first check
    initial_timer = threading.Timer(60, check_jupyter_server)  # First check after 1 minute
    initial_timer.daemon = True
    initial_timer.start()
    logger.info("Started periodic checks for Jupyter model server")

# Start the scheduled checks
if JUPYTER_MODEL_SERVER_AVAILABLE:
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
    Health check endpoint with Jupyter model server status information.
    """
    global model_status
    
    # Track execution time of the health check
    health_check_start = time.time()
    
    # Get initialization information from app state
    init_info = getattr(app.state, 'init_result', {})
    
    # Get environment information
    from .utils.initialization import detect_environment
    env_info = detect_environment()
    
    # Check Jupyter model server status
    jupyter_server_status = "unknown"
    if JUPYTER_MODEL_SERVER_AVAILABLE:
        try:
            jupyter_server_status = "running" if is_jupyter_server_running() else "stopped"
        except Exception as e:
            jupyter_server_status = f"error: {str(e)}"
    else:
        jupyter_server_status = "not available"
    
    # Free Tier: Skip disk space checks to maintain compatibility
    disk_space = {
        "note": "Disk space reporting disabled for Free Tier compatibility",
        "status": "Available disk space should be sufficient for model storage"
    }
    
    # Prepare status response
    response = {
        "status": "healthy" if jupyter_server_status == "running" else "degraded",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_info.get("environment", settings.environment),
        "model": {
            "jupyter_server": jupyter_server_status,
            "loaded": model_status["loaded"],
            "load_attempts": model_status["load_attempts"],
            "last_attempt": model_status["last_attempt_time"]
        },
        "diagnostics": {
            "runtime_info": {
                "python_version": sys.version.split()[0],
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "memory_info": {
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2) if 'psutil' in sys.modules else "N/A",
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2) if 'psutil' in sys.modules else "N/A"
                }
            },
            "initialization": init_info,
            "deployment_info": {
                "in_docker": env_info.get("in_docker", False),
                "on_render": env_info.get("on_render", False),
                "hostname": env_info.get("hostname", "unknown")
            },
            "disk_space": disk_space,
            "startup_time": app.state.startup_time if hasattr(app.state, 'startup_time') else "unknown",
            "health_check_time": round(time.time() - health_check_start, 3)
        },
        "api_version": settings.version
    }
    
    # Include debug info for developers
    if settings.debug or settings.environment != "production":
        response["debug_info"] = {
            "python_path": sys.path,
            "loaded_modules": list(sys.modules.keys())[:20],  # First 20 modules
            "environment_variables": {k: v for k, v in os.environ.items() 
                                      if not any(secret in k.lower() for secret in ['key', 'secret', 'password', 'token'])}
        }
    
    return response


@app.post("/api/download-model")
async def download_model_endpoint():
    """Endpoint is deprecated - model is now handled by Jupyter server."""
    return {
        "success": False,
        "message": "This endpoint is deprecated. The model is now handled by the Jupyter model server.",
        "jupyter_server": JUPYTER_MODEL_SERVER_AVAILABLE
    }

@app.post("/api/query", response_model=Dict[str, Any])
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def process_query(request: QueryRequest, request_obj: Request):
    """Process a query using the Jupyter model server"""
    global model_status
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request_obj)
    
    # Validate and sanitize the query
    sanitized_query = sanitize_input(request.query)
    is_valid, error_message = validate_query(sanitized_query)
    
    if not is_valid:
        logger.warning(f"Invalid query from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
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
        
        # Log the input
        logger.info(f"Model input: query_length={len(sanitized_query)}, context_length={len(context)}")
        
        # Check if Jupyter model server is available
        if not JUPYTER_MODEL_SERVER_AVAILABLE:
            logger.error("Jupyter model server not available - cannot process query")
            raise HTTPException(status_code=503, detail="AI model server not available. Please try again later.")
        
        # Set a timeout for the prediction
        try:
            # Get prediction using Jupyter model server
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                # Use Jupyter model server for prediction
                logger.info("Using Jupyter model server for prediction")
                model_input = {
                    'query_text': sanitized_query,
                    'passage_text': context
                }
                prediction_result = await predict_with_jupyter_server(model_input)
                
                # Check for errors
                if "error" in prediction_result:
                    logger.error(f"Jupyter model server prediction error: {prediction_result['error']}")
                    raise Exception(f"Jupyter model server prediction error: {prediction_result['error']}")
                
                # Convert to expected format
                prediction = {
                    'start_index': prediction_result.get('start_index', 0),
                    'end_index': prediction_result.get('end_index', 0),
                    'confidence': prediction_result.get('confidence', 0.0)
                }
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
    global model_status
    
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
    
    # Check if Jupyter model server is available
    if not JUPYTER_MODEL_SERVER_AVAILABLE:
        logger.error("Jupyter model server not available - cannot process chat message")
        raise HTTPException(status_code=503, detail="AI model server not available. Please try again later.")
    
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
            # Get prediction from the Jupyter model server
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                if JUPYTER_MODEL_SERVER_AVAILABLE:
                    # Use Jupyter model server for prediction
                    prediction = await predict_with_jupyter_server(model_input)
                else:
                    logger.error("Jupyter model server not available")
                    raise HTTPException(
                        status_code=503,
                        detail="AI model server not available. Please try again later."
                    )
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
