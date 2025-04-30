"""
Security utilities for the Backdoor AI application.
This module provides security features like headers, rate limiting, and input validation.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from fastapi import Request, Response, FastAPI, Depends, HTTPException, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from secure import SecureHeaders

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize secure headers
secure_headers = SecureHeaders(
    server=False,  # Don't expose server info
    hsts=True,     # HTTP Strict Transport Security
    xfo="DENY",    # X-Frame-Options: DENY
    xxp="1; mode=block",  # X-XSS-Protection
    content="nosniff",    # X-Content-Type-Options
    referrer="strict-origin-when-cross-origin", # Referrer Policy
    permissions_policy={  # Permissions Policy (formerly Feature Policy)
        "camera": None,
        "microphone": None,
        "geolocation": None,
    },
    cache_control="no-store,max-age=0"  # Cache Control
)

class QueryBlocklist:
    """Blocklist for sensitive or harmful query patterns."""
    
    def __init__(self):
        # Default blocklist patterns - extend as needed
        self.patterns = {
            # Generic harmful patterns
            "personal_data": [
                r"social security number",
                r"credit card",
                r"passport number",
                r"bank account",
                r"password",
                r"login credentials",
            ],
            # Harmful intent patterns
            "harmful_intent": [
                r"hack into",
                r"break into",
                r"exploit vulnerability",
                r"bypass security",
                r"steal data",
                r"illegal access",
            ],
            # Malicious code patterns
            "malicious_code": [
                r"rm -rf",
                r"format[^a-z]",
                r"del /f",
                r"deltree",
                r"drop database",
                r"drop table",
                r"exec(\s*\([^)]*\))?",
                r"eval(\s*\([^)]*\))?",
            ]
        }
        
        # Compile regex patterns for performance
        import re
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def is_blocked(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query contains blocked patterns.
        
        Args:
            query: The query to check
            
        Returns:
            Tuple of (is_blocked, reason)
        """
        query = query.lower()
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return True, f"Query matched blocked pattern category: {category}"
        
        return False, None
    
    def add_pattern(self, pattern: str, category: str = "custom"):
        """
        Add a pattern to the blocklist.
        
        Args:
            pattern: Regex pattern to block
            category: Category to assign the pattern to
        """
        import re
        
        if category not in self.patterns:
            self.patterns[category] = []
            self.compiled_patterns[category] = []
            
        self.patterns[category].append(pattern)
        self.compiled_patterns[category].append(re.compile(pattern, re.IGNORECASE))

# Initialize query blocklist
query_blocklist = QueryBlocklist()

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: The input text to sanitize
        
    Returns:
        Sanitized text
    """
    # Basic sanitization - remove control characters
    sanitized = ''.join(c for c in text if ord(c) >= 32 or c == '\n' or c == '\t')
    
    # Remove potentially harmful HTML/script tags
    import re
    sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'<.*?javascript:.*?>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized

def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a user query for security and compliance.
    
    Args:
        query: The query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty queries
    if not query or query.strip() == "":
        return False, "Query cannot be empty"
    
    # Check length limits
    if len(query) > settings.ai_response.max_response_length * 2:
        return False, f"Query exceeds maximum length of {settings.ai_response.max_response_length * 2} characters"
    
    # Check blocklist
    is_blocked, reason = query_blocklist.is_blocked(query)
    if is_blocked:
        logger.warning(f"Blocked query: {reason}")
        return False, "This query contains prohibited content and cannot be processed"
    
    return True, None

def setup_security(app: FastAPI):
    """
    Set up security features for the FastAPI application.
    
    Args:
        app: The FastAPI application
    """
    # Add trusted host middleware if configured
    if settings.security.enable_https_redirect or settings.security.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=settings.security.trusted_hosts
        )
    
    # Setup rate limiting if enabled
    if settings.api.enable_rate_limiting:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add middleware for security headers if enabled
    if settings.security.enable_security_headers:
        @app.middleware("http")
        async def set_secure_headers(request: Request, call_next):
            response = await call_next(request)
            secure_headers.apply(response)
            return response

def get_client_info(request: Request) -> Dict[str, Any]:
    """
    Get client information from the request.
    
    Args:
        request: The FastAPI request
        
    Returns:
        Dictionary with client information
    """
    headers = dict(request.headers)
    safe_headers = {
        k: v for k, v in headers.items() 
        if k.lower() not in ["authorization", "cookie"]
    }
    
    # Create a unique client ID based on IP and user agent
    ip = get_remote_address(request)
    user_agent = headers.get("user-agent", "")
    client_id = hashlib.sha256(f"{ip}:{user_agent}".encode()).hexdigest()[:12]
    
    return {
        "client_id": client_id,
        "ip": ip,
        "user_agent": user_agent,
        "referer": headers.get("referer", ""),
        "timestamp": time.time()
    }

def rate_limit_ip_and_tokens(tokens_per_minute: int = 60):
    """
    Rate limiting decorator based on client IP and token count.
    Limits both requests per minute and total tokens per minute.
    
    Args:
        tokens_per_minute: Maximum tokens allowed per minute
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        # Apply standard rate limiting
        limited_func = limiter.limit(f"{settings.api.rate_limit_calls}/minute")(func)
        
        async def wrapper(*args, **kwargs):
            # Get the original response
            response = await limited_func(*args, **kwargs)
            
            # TODO: Implement token-based rate limiting by counting tokens
            # in the request and maintaining a token counter per client
            
            return response
        
        # Copy metadata from the original function
        wrapper.__name__ = func.__name__
        return wrapper
    
    return decorator
