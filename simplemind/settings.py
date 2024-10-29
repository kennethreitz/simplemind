from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    XAI_API_KEY: str = Field(..., env="XAI_API_KEY")


settings = Settings()
