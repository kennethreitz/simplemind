from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    ollama_host_url: Optional[str] = Field(None, env="OLLAMA_HOST_URL")
    default_model: str = Field("gpt-4", env="DEFAULT_MODEL")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    @field_validator("*", mode="before")
    def check_required(cls, v, info):
        if info.field_name in info.data and info.data[info.field_name] is None:
            raise ValueError(f"{info.field_name} is required")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
