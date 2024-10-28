from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Set


class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    ollama_host_url: Optional[str] = Field(None, env="OLLAMA_HOST_URL")
    default_model: str = Field("gpt-4", env="DEFAULT_MODEL")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Map of provider names to their required environment variables
    PROVIDER_REQUIREMENTS: Dict[str, str] = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "ollama": "ollama_host_url",
    }

    @field_validator("*", mode="before")
    def check_empty_string(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def validate_provider(self, provider: str) -> bool:
        """
        Validate that the necessary API key exists for a given provider.
        Raises ValueError if the provider is not properly configured.
        """
        if provider not in self.PROVIDER_REQUIREMENTS:
            raise ValueError(f"Unknown provider: {provider}")

        required_key = self.PROVIDER_REQUIREMENTS[provider]
        if getattr(self, required_key) is None:
            raise ValueError(
                f"Missing API key for {provider}. "
                f"Please set {required_key.upper()} environment variable."
            )
        return True

    @property
    def available_providers(self) -> Set[str]:
        """Return a set of properly configured providers."""
        return {
            provider
            for provider, key in self.PROVIDER_REQUIREMENTS.items()
            if getattr(self, key) is not None
        }

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
