from typing import Optional, Union

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """The class that holds all the API keys for the application."""

    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        None, description="API key for Anthropic"
    )
    GROQ_API_KEY: Optional[SecretStr] = Field(None, description="API key for Groq")
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for OpenAI")
    OLLAMA_HOST_URL: Optional[str] = Field(
        "http://127.0.0.1:11434", description="Fully qualified host URL for Ollama"
    )
    XAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for xAI")
    DEFAULT_LLM_PROVIDER: str = Field("openai", description="The default LLM provider")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

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
