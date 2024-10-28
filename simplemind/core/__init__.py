from typing import Dict, Any, Optional
from .models import AIResponse
from ..concepts.context import Context
from ..providers.base import BaseClientProvider

from .config import settings


class SimpleMind:
    """Main class for SimpleMind functionality."""

    def __init__(self, provider: str = "openai", context: Optional[Context] = None):
        """Initialize SimpleMind with the specified provider."""
        self.provider = provider
        self.context = context or Context()
        self._client = self._get_provider()

    def _get_provider(self) -> BaseClientProvider:
        """Get the appropriate provider client."""
        from ..providers.openai import OpenAI
        from ..providers.anthropic import Anthropic

        # Initialize providers based on environment variables.
        providers = {}
        if settings.openai_api_key:
            providers.update({"openai": OpenAI})
        if settings.anthropic_api_key:
            providers.update({"anthropic": Anthropic})

        if self.provider not in providers:
            raise ValueError(
                f"Provider '{self.provider}' not supported. Available providers: {list(providers.keys())}"
            )

        return providers[self.provider](api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> AIResponse:
        """Generate a response using the configured provider."""

        return self._client.message(prompt, **kwargs)

    def create_conversation(self, initial_message: str) -> str:
        """Create a new conversation and return its ID."""

        conversation = self._client.create_conversation(initial_message)
        return conversation.id

    def send_message(self, conversation_id: str, message: str) -> AIResponse:
        """Send a message in an existing conversation."""

        return self._client.send_message(conversation_id, message)
