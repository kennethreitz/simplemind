from typing import Optional
from simplemind.core.models import Conversation, AIResponse
from simplemind.concepts.context import Context
from simplemind.providers.openai import OpenAI
from simplemind.providers.anthropic import Anthropic
import logging
from .errors import ProviderError
from .logger import logger


class Client:
    def __init__(self, api_key: str, context: Optional[Context] = None):
        self.api_key = api_key
        self.context = context or Context()
        self.providers = self._initialize_providers()

    def _initialize_providers(self):
        return {
            "openai": OpenAI(api_key=self.api_key),
            "anthropic": Anthropic(api_key=self.api_key),
        }

    def create_conversation(
        self, provider: str = "openai", context: Optional[Context] = None
    ) -> Conversation:
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not supported.")
        return self.providers[provider].create_conversation(
            initial_message="Hello!", context=self.context.model_dump()
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
