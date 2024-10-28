from typing import Optional
from simplemind.models import Conversation, AIResponse
from simplemind.concepts import Context
from simplemind.integrations.openai import OpenAI
from simplemind.integrations.anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)


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

    def create_conversation(self, provider: str = "openai") -> Conversation:
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not supported.")
        return self.providers[provider].create_conversation(
            initial_message="Hello!", context=self.context.dict()
        )

    def _handle_api_error(self, error: Exception, operation: str):
        """Handle API errors in a consistent way."""
        logger.error(f"Error during {operation}: {str(error)}")
        raise RuntimeError(f"Failed to {operation}: {str(error)}")

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
