from typing import List, Optional
import openai
from ..core.errors import AuthenticationError, ProviderError
from ..core.config import settings
from ..core.logger import logger
from .base import BaseClientProvider

DEFAULT_MODEL = "gpt-4o"


class OpenAI(BaseClientProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)
        self.login()

    def login(self) -> None:
        if not self._api_key:
            self._api_key = settings.openai_api_key
        if not self._api_key:
            raise AuthenticationError("OpenAI API key not provided")

        try:
            openai.api_key = self._api_key
            if not self.test_connection():
                raise ProviderError("Failed to connect to OpenAI API")
            logger.info("Successfully connected to OpenAI")
        except Exception as e:
            raise ProviderError(f"OpenAI initialization failed: {str(e)}")

    @property
    def available_models(self) -> List[str]:
        try:
            models = openai.models.list()
            return [model.id for model in models["data"]]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    def test_connection(self):
        """Test the connection to OpenAI API."""
        try:
            openai.models.list()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _handle_api_error(self, e: Exception) -> None:
        """Handle API errors."""
        logger.error(f"OpenAI API error: {e}")
        raise ProviderError(f"OpenAI API error: {e}")

    def add_message(self, conversation_id, message, *args, **kwargs) -> str:
        """Generate a response using the OpenAI API."""
        try:
            # Create a client instance using the API key
            client = openai.OpenAI(api_key=self._api_key)

            # Create the message for the conversation
            messages = [{"role": "user", "content": message}]

            # Use the new API syntax
            response = client.chat.completions.create(
                model=self.model, messages=messages, *args, **kwargs
            )

            return response.choices[0].message.content
        except Exception as e:
            self._handle_api_error(e)
