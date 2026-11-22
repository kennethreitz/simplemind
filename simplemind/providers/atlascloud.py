from functools import cached_property

from ..settings import settings
from .openai import OpenAI


class AtlasCloud(OpenAI):
    NAME = "atlascloud"
    DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
    BASE_URL = "https://api.atlascloud.ai/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(self.NAME)

    @cached_property
    def client(self):
        """The raw OpenAI-compatible Atlas Cloud client."""
        if not self.api_key:
            raise ValueError("Atlas Cloud API key is required")
        try:
            import openai as oa
        except ImportError as exc:
            raise ImportError(
                "Please install the `openai` package: `pip install openai`"
            ) from exc
        return oa.OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
