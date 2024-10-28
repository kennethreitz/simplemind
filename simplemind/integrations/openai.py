from typing import List, Optional
from openai import OpenAI as BaseOpenAI
from ..core.errors import AuthenticationError, ProviderError
from simplemind.core.config import settings
from ..core.logger import logger
from .base import BaseClientProvider


class OpenAI(BaseClientProvider):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)
        self.login()

    def login(self) -> None:
        if not self._api_key:
            self._api_key = settings.openai_api_key
        if not self._api_key:
            raise AuthenticationError("OpenAI API key not provided")

        try:
            self.client = BaseOpenAI(api_key=self._api_key)
            if not self.test_connection():
                raise ProviderError("Failed to connect to OpenAI API")
            logger.info("Successfully connected to OpenAI")
        except Exception as e:
            raise ProviderError(f"OpenAI initialization failed: {str(e)}")

    @property
    def available_models(self) -> List[str]:
        try:
            return [model.id for model in self.client.models.list()]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    def test_connection(self):
        """Test the connection to OpenAI API."""
        try:
            # A simple test call to verify API key works
            self.client.models.list()
            return True
        except Exception as e:
            return False

    def _handle_api_error(self, e: Exception) -> None:
        """Handle API errors."""
        logger.error(f"OpenAI API error: {e}")
        raise ProviderError(f"OpenAI API error: {e}")

    def generate_response(self, conversation) -> str:
        """Generate a response using the OpenAI API."""
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation.messages
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )

            return response.choices[0].message.content
        except Exception as e:
            self._handle_api_error(e)
