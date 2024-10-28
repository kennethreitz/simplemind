import os
from typing import Optional

from .models import Conversation, AIResponse
from ..concepts.context import Context

from .errors import ProviderError
from .logger import logger


class Client:
    def __init__(self, api_key=None):
        self.providers = {}

        # Auto-detect available API keys from environment
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            # Add other providers as needed
        }

        # Initialize providers for which we have API keys
        for provider, key in api_keys.items():
            if key:
                self.providers[provider] = self._initialize_provider(provider, key)

    def _initialize_provider(self, provider_name, api_key):
        if provider_name == "openai":
            from ..providers.openai import OpenAI

            return OpenAI(api_key)
        elif provider_name == "anthropic":
            from ..providers.anthropic import Anthropic

            return Anthropic(api_key)
        # Add other providers as needed

    def create_conversation(
        self, provider: str = "openai", context: Optional[Context] = None
    ) -> Conversation:
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not supported.")
        conversation_context = context or Context()
        return self.providers[provider].create_conversation(
            initial_message="Hello!", context=conversation_context.model_dump()
        )

    def _handle_api_error(self, error: Exception, operation: str):
        """Handle API errors in a consistent way."""
        logger.error(f"Error during {operation}: {str(error)}")
        raise ProviderError(f"Failed to {operation}: {str(error)}") from error

    def send_message(
        self, conversation: Conversation, message: str, provider: str = "openai"
    ) -> AIResponse:
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not supported.")
        try:
            return self.providers[provider].send_message(conversation.id, message)
        except Exception as e:
            self._handle_api_error(e, "send message")

    @property
    def available_models(self):
        available = {}
        for name, provider in self.providers.items():
            available[name] = provider.available_models
        return available
