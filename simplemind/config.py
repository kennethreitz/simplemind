from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_model: str = "gpt-4"

    class Config:
        env_file = ".env"


settings = Settings()
