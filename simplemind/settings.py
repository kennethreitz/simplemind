from typing import Optional, Union

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    """The class that holds all the logging settings for the application."""

    is_enabled: bool = Field(False, description="Enable logging")
    model_config = SettingsConfigDict(extra="forbid")

    def enable_logfire(self, **kwargs) -> None:
        """Enable logging for the application."""
        # adding imports here to avoid forced dependencies
        try:
            import logfire
            from logging import basicConfig
        except ImportError as e:
            raise ImportError(
                "To enable logging, please install logfire: `pip install logfire`"
            ) from e

        self.is_enabled = True
        logfire.configure(**kwargs)
        basicConfig(handlers=[logfire.LogfireLoggingHandler()])

        try:
            logfire.configure(**kwargs)
            basicConfig(handlers=[logfire.LogfireLoggingHandler()])
        except Exception as e:
            self.is_enabled = False  # Reset flag on failure
            raise RuntimeError("Failed to configure logging") from e

    def disable_logfire(self) -> None:
        """Disable logging for the application."""
        self.is_enabled = False


class Settings(BaseSettings):
    """The class that holds all the API keys for the application."""

    AMAZON_PROFILE_NAME: Optional[str] = Field(
        "default", description="AWS Named Profile"
    )
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        None, description="API key for Anthropic"
    )
    GROQ_API_KEY: Optional[SecretStr] = Field(None, description="API key for Groq")
    GEMINI_API_KEY: Optional[SecretStr] = Field(None, description="API key for Gemini")
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for OpenAI")
    OLLAMA_HOST_URL: Optional[str] = Field(
        "http://127.0.0.1:11434", description="Fully qualified host URL for Ollama"
    )
    XAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for xAI")
    DEFAULT_LLM_PROVIDER: str = Field("openai", description="The default LLM provider")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )
    logging: LoggingConfig = LoggingConfig()

    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str) -> Optional[str]:
        """Convert empty strings to None for optional fields."""
        if v == "":
            return None
        return v

    def get_api_key(self, provider: str) -> Union[str, None]:
        """
        Safely get API key for a specific provider.
        Returns the key as a string or None if not set.
        """
        key = getattr(self, f"{provider.upper()}_API_KEY", None)
        return key.get_secret_value() if key else None


settings = Settings()
