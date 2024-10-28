from typing import List, Optional
from openai import OpenAI as BaseOpenAI
from ..core.errors import AuthenticationError, ProviderError
from ..core.config import settings
from ..core.logger import logger
from .base import BaseClientProvider

DEFAULT_MODEL = "gpt-4"


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

    def generate_response(self, conversation, **kwargs) -> str:
        """Generate a response using the OpenAI API."""
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation.messages
            ]

            # Ensure we're using a valid model name, not the API key
            if not isinstance(self.model, str) or self.model.startswith("sk-"):
                logger.warning(
                    f"Invalid model name detected. Falling back to {DEFAULT_MODEL!r}"
                )
                model_name = DEFAULT_MODEL
            else:
                model_name = self.model

            r = self.client.chat.completions.create(
                model=model_name,  # Use the validated model name
                messages=messages,
                **kwargs,
            )
            return r
        except Exception as e:
            self._handle_api_error(e)

    def generate_text(self, conversation, **kwargs) -> str:
        """Generate a text response using the OpenAI API."""
        return self.generate_response(conversation, **kwargs).choices[0].message.content
