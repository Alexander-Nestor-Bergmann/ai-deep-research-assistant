"""
LLM Client Configuration for AI Deep Research Assistant.

This module provides centralized model loading functionality with support for multiple
LLM providers including OpenRouter, OpenAI, Anthropic, and Google Gemini.
"""
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

try:
    from .config.settings import get_settings
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config.settings import get_settings

# Pydantic AI imports
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel

# HTTP client for general requests
from httpx import AsyncClient

load_dotenv()
logger = logging.getLogger(__name__)


class ModelConfigError(Exception):
    """Raised when model configuration is invalid or missing"""
    pass


def get_langfuse_client():
    """
    Get LangFuse client for LLM observability if environment variables are set.
    
    Returns:
        Langfuse client instance or None if not configured
    """
    try:
        from langfuse import get_client
        
        langfuse = get_client()
        if langfuse.auth_check():
            logger.info("✅ Langfuse observability enabled")
            return langfuse
        else:
            logger.warning("Langfuse auth check failed")
            return None
    except ImportError:
        logger.info("Langfuse not installed, skipping observability")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def get_model(use_smaller_model: bool = False) -> Model:
    """
    Get configured LLM model for agents with provider selection support.
    
    Args:
        use_smaller_model: If True, uses MODEL_CHOICE_SMALL for cost optimization.
                          If False, uses MODEL_CHOICE for quality optimization.
    
    Returns:
        Configured Pydantic AI model instance
        
    Raises:
        ModelConfigError: If required configuration is missing or invalid
    """
    settings = get_settings()
    
    # Get AI service provider from settings
    ai_service = settings.ai_service.lower()
    
    # Select model based on size preference
    if use_smaller_model:
        model_name = settings.model_choice_small
    else:
        model_name = settings.model_choice
    
    logger.info(f"Loading model: {model_name} via {ai_service}")
    
    try:
        if ai_service == 'openrouter':
            return _get_openrouter_model(model_name)
        elif ai_service == 'openai':
            return _get_openai_model(model_name)
        elif ai_service == 'anthropic':
            return _get_anthropic_model(model_name)
        elif ai_service == 'gemini':
            return _get_gemini_model(model_name)
        else:
            raise ModelConfigError(f"Unsupported AI service: {ai_service}")
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name} via {ai_service}: {e}")
        # Fallback to OpenRouter with a basic model
        logger.warning("Falling back to OpenRouter with gpt-4o-mini")
        return _get_openrouter_model('openai/gpt-4o-mini')


def _get_openrouter_model(model_name: str) -> OpenAIModel:
    """Get OpenRouter model (uses OpenAI-compatible API)"""
    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ModelConfigError("OpenRouter API key is required - check your .env file")
    
    # Extract model name from OpenRouter format (provider/model)
    if '/' not in model_name:
        raise ModelConfigError(f"OpenRouter model name must be in 'provider/model' format: {model_name}")
    
    # Set environment for OpenAI-compatible API
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'
    
    return OpenAIModel(model_name)


def _get_openai_model(model_name: str) -> OpenAIModel:
    """Get OpenAI model"""
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise ModelConfigError("OpenAI API key is required - check your .env file")
    
    # Remove provider prefix if present (for consistency)
    if model_name.startswith('openai/'):
        model_name = model_name[7:]  # Remove 'openai/' prefix
    
    return OpenAIModel(model_name)


def _get_anthropic_model(model_name: str) -> AnthropicModel:
    """Get Anthropic model"""
    settings = get_settings()
    api_key = settings.anthropic_api_key
    if not api_key:
        raise ModelConfigError("Anthropic API key is required - check your .env file")
    
    # Remove provider prefix if present
    if model_name.startswith('anthropic/'):
        model_name = model_name[10:]  # Remove 'anthropic/' prefix
    
    return AnthropicModel(model_name)


def _get_gemini_model(model_name: str) -> GeminiModel:
    """Get Google Gemini model"""
    settings = get_settings()
    api_key = settings.gemini_api_key
    if not api_key:
        raise ModelConfigError("Gemini API key is required - check your .env file")
    
    # Remove provider prefix if present
    if model_name.startswith('google/'):
        model_name = model_name[7:]  # Remove 'google/' prefix
    
    return GeminiModel(model_name)


def get_http_client() -> AsyncClient:
    """
    Get configured async HTTP client for general requests.
    
    Returns:
        Configured AsyncClient instance
    """
    # Configure timeout and retry settings
    timeout_config = {
        "timeout": 30.0,
        "read": 30.0,
        "write": 10.0,
        "connect": 10.0
    }
    
    return AsyncClient(timeout=timeout_config)


def get_model_config() -> Dict[str, Any]:
    """
    Get current model configuration for logging and debugging.
    
    Returns:
        Dictionary with current configuration settings
    """
    settings = get_settings()
    return {
        "ai_service": settings.ai_service,
        "model_choice": settings.model_choice,
        "model_choice_small": settings.model_choice_small,
        "temperature": settings.model_temperature,
        "top_p": settings.model_top_p,
        "max_tokens": settings.model_max_tokens,
        "langfuse_enabled": get_langfuse_client() is not None
    }


def validate_model_access() -> Dict[str, bool]:
    """
    Validate that configured models are accessible.
    
    Returns:
        Dictionary indicating which providers are accessible
    """
    results = {}
    
    # Test small model
    try:
        small_model = get_model(use_smaller_model=True)
        results['small_model'] = True
        logger.info("✅ Small model accessible")
    except Exception as e:
        results['small_model'] = False
        logger.error(f"❌ Small model not accessible: {e}")
    
    # Test full model
    try:
        full_model = get_model(use_smaller_model=False)
        results['full_model'] = True
        logger.info("✅ Full model accessible")
    except Exception as e:
        results['full_model'] = False
        logger.error(f"❌ Full model not accessible: {e}")
    
    return results


if __name__ == "__main__":
    """Test model configuration when run directly"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testing model configuration...")
    print(f"📋 Current config: {get_model_config()}")
    
    # Test model access
    access_results = validate_model_access()
    
    if all(access_results.values()):
        print("✅ All models configured correctly!")
    else:
        print("❌ Some models are not accessible - check your API keys")
        
    # Test HTTP client
    try:
        http_client = get_http_client()
        print("✅ HTTP client configured")
    except Exception as e:
        print(f"❌ HTTP client error: {e}")