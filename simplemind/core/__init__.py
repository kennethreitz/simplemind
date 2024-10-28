from typing import Dict, Any, Optional
from .models import AIResponse
from ..concepts.context import Context
from ..providers.base import BaseClientProvider


class SimpleMind:
    """Main class for SimpleMind functionality."""

    def __init__(
        self, api_key: str, provider: str = "openai", context: Optional[Context] = None
    ):
        """Initialize SimpleMind with the specified provider."""
        self.api_key = api_key
        self.provider = provider
        self.context = context or Context()
        self._client = self._get_provider()

    def _get_provider(self) -> BaseClientProvider:
        """Get the appropriate provider client."""
        from .integrations.openai import OpenAI
        from .integrations.anthropic import Anthropic

        providers = {"openai": OpenAI, "anthropic": Anthropic}

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
