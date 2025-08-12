"""
Configuration settings for AI Deep Research Assistant.

This module uses Pydantic Settings for environment variable loading and validation
with comprehensive defaults and type checking.
"""

from typing import Optional, Dict, Any
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIService(str, Enum):
    """Available AI service providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LogFormat(str, Enum):
    """Available log formats."""

    JSON = "json"
    TEXT = "text"


class LogLevel(str, Enum):
    """Available log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ResearchDepth(str, Enum):
    """Available research depth levels."""

    SURFACE = "surface"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class Settings(BaseSettings):
    """
    Application settings with environment variable loading and validation.

    All settings can be configured via environment variables with the same name
    (case insensitive). See .env.example for all available options.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # =============================================================================
    # WEB SEARCH CONFIGURATION
    # =============================================================================
    brave_api_key: str = Field(
        default="", description="Brave Search API key for web research"
    )

    # =============================================================================
    # LLM PROVIDER CONFIGURATION
    # =============================================================================
    ai_service: AIService = Field(
        default=AIService.OPENROUTER, description="LLM provider service"
    )

    @field_validator("ai_service", mode="before")
    @classmethod
    def parse_ai_service(cls, value):
        """Strip comments from AI service environment variable."""
        if isinstance(value, str):
            # Strip inline comments
            clean_value = value.split("#")[0].strip()
            return clean_value
        return value

    # Provider API Keys
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    model_choice_small: str = Field(
        default="openai/gpt-4o-mini",
        description="Small model for guardrail agent (cost optimization)",
    )

    model_choice: str = Field(
        default="anthropic/claude-3-5-sonnet-20241022",
        description="Full model for research and synthesis (quality optimization)",
    )

    model_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for response creativity",
    )

    model_top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Model top-p for nucleus sampling"
    )

    model_max_tokens: int = Field(
        default=4096, ge=100, le=32000, description="Maximum tokens per model response"
    )

    # =============================================================================
    # PERFORMANCE & CACHING CONFIGURATION
    # =============================================================================
    cache_ttl_minutes: int = Field(
        default=15, ge=1, le=1440, description="Cache TTL in minutes"
    )

    cache_max_size: int = Field(
        default=1000, ge=10, le=10000, description="Maximum cache entries"
    )

    brave_rate_limit_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between Brave API calls in seconds",
    )

    @field_validator("brave_rate_limit_delay", mode="before")
    @classmethod
    def parse_brave_rate_limit_delay(cls, value):
        """Strip comments from brave rate limit delay environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip()
            return float(clean_value)
        return value

    max_concurrent_requests: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent API requests"
    )

    first_token_timeout: int = Field(
        default=2, ge=1, le=30, description="Timeout for first token in seconds"
    )

    @field_validator("first_token_timeout", mode="before")
    @classmethod
    def parse_first_token_timeout(cls, value):
        """Strip comments from first token timeout environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip()
            return int(clean_value)
        return value

    end_to_end_timeout: int = Field(
        default=15, ge=5, le=120, description="Total request timeout in seconds"
    )

    @field_validator("end_to_end_timeout", mode="before")
    @classmethod
    def parse_end_to_end_timeout(cls, value):
        """Strip comments from end to end timeout environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip()
            return int(clean_value)
        return value

    # =============================================================================
    # OBSERVABILITY & MONITORING
    # =============================================================================
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )

    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")

    @field_validator("log_level", mode="before")
    @classmethod
    def parse_log_level(cls, value):
        """Strip comments from log level environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip().upper()
            return clean_value
        return value

    log_format: LogFormat = Field(
        default=LogFormat.TEXT, description="Log format style"
    )

    @field_validator("log_format", mode="before")
    @classmethod
    def parse_log_format(cls, value):
        """Strip comments from log format environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip().lower()
            return clean_value
        return value

    # =============================================================================
    # RESEARCH CONFIGURATION
    # =============================================================================
    quick_mode: bool = Field(
        default=False,
        description="Enable quick mode for faster but less comprehensive research",
    )

    max_search_results: int = Field(
        default=8, ge=1, le=20, description="Maximum search results per query"
    )

    max_citations_per_claim: int = Field(
        default=3, ge=1, le=10, description="Maximum citations per research claim"
    )

    default_research_depth: ResearchDepth = Field(
        default=ResearchDepth.COMPREHENSIVE, description="Default research depth level"
    )

    @field_validator("default_research_depth", mode="before")
    @classmethod
    def parse_default_research_depth(cls, value):
        """Strip comments from default research depth environment variable."""
        if isinstance(value, str):
            clean_value = value.split("#")[0].strip().lower()
            return clean_value
        return value

    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for including results",
    )

    synthesis_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for synthesis completion",
    )

    # =============================================================================
    # CLI CONFIGURATION
    # =============================================================================
    enable_rich_formatting: bool = Field(
        default=True, description="Enable rich console formatting"
    )

    show_progress_bars: bool = Field(
        default=True, description="Show progress bars during execution"
    )

    show_citation_previews: bool = Field(
        default=True, description="Show citation previews in CLI"
    )

    enable_follow_up_questions: bool = Field(
        default=True, description="Enable follow-up question suggestions"
    )

    max_conversation_history: int = Field(
        default=10, ge=1, le=50, description="Maximum conversation history entries"
    )

    # =============================================================================
    # DEVELOPMENT SETTINGS
    # =============================================================================
    debug: bool = Field(default=False, description="Enable debug mode")

    verbose_logging: bool = Field(default=False, description="Enable verbose logging")

    mock_brave_api: bool = Field(
        default=False, description="Use mock Brave API for testing"
    )

    test_mode: bool = Field(default=False, description="Enable test mode")

    @field_validator(
        "brave_api_key",
        "openrouter_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        mode="before",
    )
    @classmethod
    def strip_api_keys(cls, v):
        """Strip whitespace from API keys"""
        return v.strip() if isinstance(v, str) else v

    @field_validator("model_choice_small", "model_choice")
    @classmethod
    def validate_model_names(cls, v):
        """Validate model name format"""
        if not v or not isinstance(v, str):
            raise ValueError("Model name cannot be empty")
        return v.strip()

    def get_api_key_for_service(self) -> Optional[str]:
        """Get the API key for the configured AI service"""
        service_key_mapping = {
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }

        return service_key_mapping.get(self.ai_service, "")

    def is_service_configured(self) -> bool:
        """Check if the current AI service is properly configured"""
        api_key = self.get_api_key_for_service()
        return bool(api_key and api_key.strip())

    def is_web_search_configured(self) -> bool:
        """Check if web search is properly configured"""
        return bool(self.brave_api_key and self.brave_api_key.strip())

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary"""
        return {
            "service": self.ai_service,
            "model_small": self.model_choice_small,
            "model_full": self.model_choice,
            "temperature": self.model_temperature,
            "top_p": self.model_top_p,
            "max_tokens": self.model_max_tokens,
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration as dictionary"""
        return {"ttl_minutes": self.cache_ttl_minutes, "max_size": self.cache_max_size}

    def get_research_config(self) -> Dict[str, Any]:
        """Get research configuration as dictionary"""
        return {
            "max_search_results": self.max_search_results,
            "max_citations_per_claim": self.max_citations_per_claim,
            "research_depth": self.default_research_depth,
            "min_confidence": self.min_confidence_threshold,
            "synthesis_confidence": self.synthesis_confidence_threshold,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern).

    Returns:
        Settings instance loaded from environment
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment (useful for testing).

    Returns:
        Reloaded Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings


if __name__ == "__main__":
    """Test configuration loading when run directly"""
    import json

    settings = get_settings()

    print("🔧 Current Configuration:")
    print(f"  AI Service: {settings.ai_service}")
    print(f"  Model (Small): {settings.model_choice_small}")
    print(f"  Model (Full): {settings.model_choice}")
    print(f"  Service Configured: {settings.is_service_configured()}")
    print(f"  Web Search Configured: {settings.is_web_search_configured()}")

    print("\n📋 Model Config:")
    print(json.dumps(settings.get_model_config(), indent=2))

    print("\n💾 Cache Config:")
    print(json.dumps(settings.get_cache_config(), indent=2))

    print("\n🔍 Research Config:")
    print(json.dumps(settings.get_research_config(), indent=2))
