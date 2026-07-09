from functools import cached_property

from ..settings import settings
from .anthropic import Anthropic
from .openai import OpenAI


class _MiniMaxConfig:
    MODEL_IDS = ("MiniMax-M3", "MiniMax-M2.7")
    ENDPOINTS = {
        "global_en": {
            "openai": "https://api.minimax.io/v1",
            "anthropic": "https://api.minimax.io/anthropic/v1",
        },
        "cn_zh": {
            "openai": "https://api.minimaxi.com/v1",
            "anthropic": "https://api.minimaxi.com/anthropic/v1",
        },
    }

    def _configure(self, api_key: str | None, region: str | None) -> None:
        self.api_key = api_key or settings.get_api_key("minimax")
        self.region = region or settings.MINIMAX_API_REGION
        if self.region not in self.ENDPOINTS:
            regions = ", ".join(self.ENDPOINTS)
            raise ValueError(f"MiniMax API region must be one of: {regions}")


class MiniMax(_MiniMaxConfig, OpenAI):
    NAME = "minimax"
    DEFAULT_MODEL = "MiniMax-M3"

    def __init__(self, api_key: str | None = None, region: str | None = None):
        self._configure(api_key, region)

    @cached_property
    def client(self):
        """The raw OpenAI-compatible MiniMax client."""
        if not self.api_key:
            raise ValueError("MiniMax API key is required")
        try:
            import openai as oa
        except ImportError as exc:
            raise ImportError(
                "Please install the `openai` package: `pip install openai`"
            ) from exc
        return oa.OpenAI(
            api_key=self.api_key,
            base_url=self.ENDPOINTS[self.region]["openai"],
        )


class MiniMaxAnthropic(_MiniMaxConfig, Anthropic):
    NAME = "minimax-anthropic"
    DEFAULT_MODEL = "MiniMax-M3"

    def __init__(self, api_key: str | None = None, region: str | None = None):
        self._configure(api_key, region)

    @cached_property
    def client(self):
        """The raw Anthropic-compatible MiniMax client."""
        if not self.api_key:
            raise ValueError("MiniMax API key is required")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Please install the `anthropic` package: `pip install anthropic`"
            ) from exc
        return anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.ENDPOINTS[self.region]["anthropic"],
        )
