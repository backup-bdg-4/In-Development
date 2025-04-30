"""
Utilities for handling and improving AI responses.
This module provides functionality to enhance the model outputs for various types of queries.
"""

import re
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import uuid
import nltk
from nltk.tokenize import sent_tokenize
import random
from langdetect import detect, LangDetectException

from ..config import settings
from .code_utils import contains_code, extract_code_blocks, is_code_question, format_code_for_response

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK resources: {e}")

# Response templates for different intents
RESPONSE_TEMPLATES = {
    "greeting": [
        "Hello! I'm Backdoor AI. How can I help you today?",
        "Hi there! I'm here to assist you with any questions you might have.",
        "Greetings! I'm Backdoor AI, ready to help with your questions.",
        "Hello! How can I assist you today?",
    ],
    "greeting_follow_up": [
        "Is there something specific you'd like to know or discuss?",
        "What can I help you with today?",
        "Do you have a question I can assist with?",
        "How can I be of assistance to you?"
    ],
    "farewell": [
        "Goodbye! Feel free to ask if you have any more questions.",
        "Take care! I'm here if you need further assistance.",
        "Farewell! Don't hesitate to return if you need more help.",
        "Goodbye! Have a great day."
    ],
    "clarification": [
        "I'm not sure I understand. Could you please provide more details?",
        "I'd like to help, but I need a bit more information. Could you elaborate?",
        "Could you please clarify what you're asking?",
        "I'm sorry, but I need more context to properly answer your question."
    ],
    "error": [
        "I apologize, but I encountered an error processing your request. Could you try rephrasing your question?",
        "I'm sorry, something went wrong while generating a response. Could we try a different approach?",
        "Unfortunately, I'm having trouble processing that request. Could you try asking in a different way?",
        "An error occurred while processing your question. Let's try a different approach."
    ],
    "unsupported": [
        "I apologize, but that's outside my capabilities.",
        "I'm afraid I can't assist with that particular request.",
        "That's beyond what I'm currently able to help with.",
        "I'm not equipped to handle that type of request."
    ]
}

# Response enhancers that can be applied to improve responses
def add_citations(response: str, sources: List[Dict[str, Any]]) -> str:
    """
    Add citations to a response based on search results.
    
    Args:
        response: The AI-generated response
        sources: List of search results used as sources
        
    Returns:
        Enhanced response with citations
    """
    if not sources:
        return response
        
    # Already has citations?
    if re.search(r'\[\d+\]', response):
        return response
        
    # Add citation numbers to the response
    for i, source in enumerate(sources, 1):
        if "title" in source and source["title"] in response:
            # Replace with citation
            response = response.replace(
                source["title"], 
                f"{source['title']} [{i}]", 
                1
            )
    
    # Append source list
    response += "\n\n**Sources:**\n"
    for i, source in enumerate(sources, 1):
        response += f"[{i}] {source.get('title', 'Untitled')} - {source.get('url', '')}\n"
    
    return response

def add_confidence_level(response: str, confidence: float) -> str:
    """
    Add a confidence indicator to the response.
    
    Args:
        response: The AI-generated response
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Response with confidence indicator
    """
    if confidence >= 0.9:
        indicator = "I'm highly confident in this answer."
    elif confidence >= 0.7:
        indicator = "I'm fairly confident in this answer."
    elif confidence >= 0.5:
        indicator = "I'm moderately confident in this answer."
    else:
        indicator = "I'm not very confident in this answer. You might want to verify this information."
    
    # Add indicator to the end of the response
    if not response.endswith(indicator):
        response = response.rstrip() + "\n\n" + indicator
    
    return response

def format_code_in_response(response: str) -> str:
    """
    Ensure code blocks in responses are properly formatted.
    
    Args:
        response: The AI-generated response
        
    Returns:
        Response with properly formatted code blocks
    """
    # Check if the response already has markdown code blocks
    if "```" in response:
        return response
    
    # Try to identify code segments that aren't properly marked
    code_pattern = r"(?:^|\n)((?:import |def |class |function |if |for |while |const |var |let |public |private ).*(?:\n[ \t]+.*){2,})"
    
    def replace_code_block(match):
        code = match.group(1)
        # Simple language detection based on keywords
        language = ""
        if "import " in code or "def " in code:
            language = "python"
        elif "function " in code or "const " in code or "let " in code:
            language = "javascript"
        elif "public " in code or "private " in code:
            language = "java"
        
        return f"\n```{language}\n{code}\n```\n"
    
    formatted_response = re.sub(code_pattern, replace_code_block, response)
    return formatted_response

def enhance_response(
    response: str, 
    query: str, 
    intent: str = "general_query",
    search_results: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None
) -> str:
    """
    Enhance a response with various improvements.
    
    Args:
        response: The raw AI-generated response
        query: The original user query
        intent: The detected intent
        search_results: Any search results used
        confidence: Confidence score if available
        
    Returns:
        Enhanced response
    """
    # Prevent empty responses
    if not response or response.strip() == "":
        return settings.ai_response.fallback_responses["default"]
    
    # Format the response
    enhanced = response.strip()
    
    # Handle code-related questions
    if is_code_question(query) or contains_code(enhanced):
        enhanced = format_code_in_response(enhanced)
    
    # Add citations for web search results
    if search_results:
        enhanced = add_citations(enhanced, search_results)
    
    # Add confidence level if provided and appropriate
    if confidence is not None and not is_code_question(query):
        enhanced = add_confidence_level(enhanced, confidence)
    
    return enhanced

def get_template_response(intent: str) -> str:
    """
    Get a template response for a specific intent.
    
    Args:
        intent: The detected intent
        
    Returns:
        A template response string
    """
    intent_key = intent.lower()
    
    if intent_key in RESPONSE_TEMPLATES:
        templates = RESPONSE_TEMPLATES[intent_key]
        return random.choice(templates)
    
    # If no specific templates, return a generic one
    return "I'll do my best to help you with that."

def categorize_query(query: str) -> Dict[str, Any]:
    """
    Categorize a user query to determine how to respond.
    
    Args:
        query: The user's query
        
    Returns:
        Dictionary with query categorization
    """
    query_lower = query.lower().strip()
    
    # Basic categorization
    categorization = {
        "is_code_question": is_code_question(query),
        "has_code": contains_code(query),
        "language": None,
        "intent": "general_query"
    }
    
    # Detect greetings
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if any(greeting in query_lower for greeting in greetings) and len(query_lower.split()) < 5:
        categorization["intent"] = "greeting"
    
    # Detect farewells
    farewells = ["goodbye", "bye", "see you", "farewell", "thanks", "thank you"]
    if any(farewell in query_lower for farewell in farewells) and len(query_lower.split()) < 5:
        categorization["intent"] = "farewell"
    
    # Check for code
    if categorization["has_code"]:
        code_blocks = extract_code_blocks(query)
        if code_blocks and code_blocks[0][1]:
            categorization["language"] = code_blocks[0][1]
    
    return categorization

def log_query_response(query: str, response: str, metadata: Dict[str, Any] = None):
    """
    Log a query and its response for monitoring and improvement.
    
    Args:
        query: The user's query
        response: The AI's response
        metadata: Additional metadata about the interaction
    """
    if not settings.logging.log_queries:
        return
        
    try:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response_summary": response[:100] + "..." if len(response) > 100 else response,
            "metadata": metadata or {}
        }
        
        # In a production system, you would log to a database or monitoring system
        # For now, just log to the application log
        if settings.logging.json_format:
            logger.info(f"QUERY_LOG: {json.dumps(log_entry)}")
        else:
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response[:100]}...")
            if metadata:
                logger.info(f"Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")
