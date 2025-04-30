"""
Configuration settings for the Backdoor AI application.
This file contains settings for the model, API, security, and other aspects of the application.
Settings can be overridden by environment variables.
"""

import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import yaml
import logging
from pathlib import Path

# Configure logging for the config module
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = os.environ.get("CONFIG_FILE", os.path.join(BASE_DIR, "config.yaml"))

class ModelSettings(BaseModel):
    """Settings for the CoreML model."""
    name: str = "BERTSQUADFP16"
    model_path: Optional[str] = None  # Will be set based on MODEL_DATA_PATH if None
    max_context_length: int = Field(default=1024, ge=100, le=10000)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    enable_batching: bool = False
    batch_size: int = Field(default=1, ge=1, le=16)
    
    # Model-specific settings
    supported_languages: List[str] = ["en"]  # Languages the model supports
    code_languages: List[str] = [
        "python", "javascript", "typescript", "java", "c", "c++", "c#", 
        "go", "rust", "ruby", "php", "swift", "kotlin", "html", "css", 
        "shell", "sql", "json", "xml", "yaml"
    ]
    
    # Runtime settings - these can be tuned for performance
    preload_model: bool = True
    unload_after_idle_minutes: int = Field(default=60, ge=5)  # Unload model after idle time to save memory
    predict_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)

class APISettings(BaseModel):
    """Settings for the FastAPI server."""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False
    cors_origins: List[str] = ["*"]
    max_request_size_mb: float = Field(default=10.0, ge=1.0, le=100.0)
    
    # API rate limiting
    enable_rate_limiting: bool = True
    rate_limit_calls: int = Field(default=60, ge=1)  # requests per minute
    rate_limit_period: int = Field(default=60, ge=1)  # period in seconds

class WebSearchSettings(BaseModel):
    """Settings for web search functionality."""
    enabled: bool = True
    search_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    max_results: int = Field(default=5, ge=1, le=20)
    max_tokens_per_result: int = Field(default=200, ge=50, le=1000)
    user_agent: str = "Mozilla/5.0 (compatible; BackdoorAI/1.0; +https://github.com/backup-bdg-4)"
    search_providers: List[str] = ["duckduckgo"]  # Supported: duckduckgo, google, bing
    safe_search: bool = True

class CacheSettings(BaseModel):
    """Settings for caching."""
    enabled: bool = True
    type: str = "memory"  # memory, redis
    ttl_seconds: int = Field(default=3600, ge=60)  # 1 hour
    redis_url: Optional[str] = None  # Used if type is redis
    max_memory_mb: int = Field(default=100, ge=10, le=1000)  # For memory cache

class LoggingSettings(BaseModel):
    """Settings for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: Optional[str] = None
    json_format: bool = False  # Structured logging in JSON format
    log_queries: bool = True  # Log user queries and responses

class SecuritySettings(BaseModel):
    """Settings for security."""
    enable_https_redirect: bool = False
    enable_security_headers: bool = True
    enable_cors: bool = True
    trusted_hosts: List[str] = ["localhost", "127.0.0.1"]
    
    # JWT settings (if authentication is implemented)
    enable_auth: bool = False
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = Field(default=60, ge=1, le=10080)  # 7 days max

class AIResponseSettings(BaseModel):
    """Settings for AI response handling."""
    min_response_length: int = Field(default=10, ge=1)
    max_response_length: int = Field(default=2000, ge=100, le=10000)
    default_context: str = "I'll try to provide a helpful and accurate response based on my knowledge."
    enable_web_search_by_default: bool = True
    fallback_responses: Dict[str, str] = {
        "default": "I don't have enough information to answer that question.",
        "coding": "I can help with coding questions, but I need more details to provide a specific answer.",
        "sensitive": "I'm not able to provide information on that topic.",
        "error": "I encountered an error processing your request. Please try again or rephrase your question."
    }
    enable_intent_detection: bool = True
    max_search_queries_per_conversation: int = Field(default=5, ge=1, le=20)

class Settings(BaseModel):
    """Main settings container."""
    app_name: str = "Backdoor AI"
    environment: str = "development"  # development, testing, production
    debug: bool = Field(default=False)
    version: str = "1.0.0"
    
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    web_search: WebSearchSettings = WebSearchSettings()
    cache: CacheSettings = CacheSettings()
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    ai_response: AIResponseSettings = AIResponseSettings()

def load_settings_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load settings from a YAML file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {file_path}: {str(e)}")
        return {}

def load_settings_from_env() -> Dict[str, Any]:
    """Load settings from environment variables with prefix BACKDOOR_."""
    env_settings = {}
    prefix = "BACKDOOR_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert BACKDOOR_MODEL_MAX_CONTEXT_LENGTH to model.max_context_length
            setting_path = key[len(prefix):].lower().split('_')
            
            # Convert string value to appropriate type (basic conversion)
            if value.lower() == 'true':
                typed_value = True
            elif value.lower() == 'false':
                typed_value = False
            elif value.isdigit():
                typed_value = int(value)
            elif value.replace('.', '', 1).isdigit():
                typed_value = float(value)
            else:
                typed_value = value
            
            # Build nested dict structure
            current = env_settings
            for i, part in enumerate(setting_path):
                if i == len(setting_path) - 1:
                    current[part] = typed_value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
    
    return env_settings

def get_settings() -> Settings:
    """Get settings with values from YAML and environment variables."""
    # Start with default settings
    settings_dict = {}
    
    # Update with values from YAML file
    settings_dict.update(load_settings_from_yaml(CONFIG_FILE))
    
    # Update with values from environment variables (highest priority)
    env_settings = load_settings_from_env()
    
    # Deep merge env_settings into settings_dict
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    
    deep_update(settings_dict, env_settings)
    
    # Special handling for model path - check multiple locations
    if not settings_dict.get('model', {}).get('model_path'):
        # Import here to avoid circular imports
        from pathlib import Path
        
        # Define all possible model locations in order of preference
        possible_locations = []
        
        # 1. Check environment variable (highest priority)
        model_data_path = os.environ.get('MODEL_DATA_PATH')
        if model_data_path:
            possible_locations.append(os.path.join(model_data_path, "BERTSQUADFP16.mlmodel"))
        
        # 2. Check in app/model directory (standard location)
        possible_locations.append(os.path.join(os.path.dirname(__file__), "model", "BERTSQUADFP16.mlmodel"))
        
        # 3. Check relative to backend directory (where GitHub Action places it)
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        possible_locations.append(os.path.join(backend_dir, "BERTSQUADFP16.mlmodel"))
        
        # 4. Check for absolute /app paths (Docker container)
        possible_locations.append("/app/app/model/BERTSQUADFP16.mlmodel")
        possible_locations.append("/app/BERTSQUADFP16.mlmodel")
        
        # 5. Check the tmp directory (Render deployment)
        possible_locations.append("/tmp/model/BERTSQUADFP16.mlmodel")
        
        # Use the first location that exists, or the preferred default
        model_path = None
        for location in possible_locations:
            if os.path.exists(location):
                logger.info(f"Found model at: {location}")
                model_path = location
                break
        
        # If no existing model is found, use the preferred location
        if not model_path:
            model_path = possible_locations[0] if possible_locations else os.path.join(
                os.path.dirname(__file__), "model", "BERTSQUADFP16.mlmodel"
            )
            logger.info(f"No existing model found, will use path: {model_path}")
        
        # Update the settings dictionary
        if 'model' not in settings_dict:
            settings_dict['model'] = {}
        settings_dict['model']['model_path'] = model_path
    
    # Create Settings object
    try:
        return Settings(**settings_dict)
    except Exception as e:
        logger.error(f"Error creating settings: {str(e)}")
        # Fall back to default settings
        return Settings()

# Create settings instance
settings = get_settings()

# Configure root logger based on settings
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format=settings.logging.format,
    filename=settings.logging.log_file if settings.logging.log_to_file else None
)

# Export settings
__all__ = ['settings', 'get_settings', 'Settings']
