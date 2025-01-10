import os
from functools import cached_property

from .openai import OpenAI


class Deepseek(OpenAI):
    NAME = "deepseek"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key=api_key)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.endpoint = "https://api.deepseek.com/v1"

    @cached_property
    def client(self):
        """The raw OpenAI client."""
        if not self.api_key:
            raise ValueError("DEEPSEEK API key is required")
        try:
            import openai as oa
        except ImportError as exc:
            raise ImportError(
                "Please install the `openai` package: `pip install openai`"
            ) from exc
        return oa.OpenAI(api_key=self.api_key, base_url=self.endpoint)
