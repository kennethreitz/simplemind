from functools import cached_property
from typing import Type, TypeVar

import google.generativeai as genai
import instructor
from pydantic import BaseModel

from ..settings import settings
from ._base import BaseProvider

PROVIDER_NAME = "gemini"
DEFAULT_MODEL = "models/gemini-1.5-flash-latest"

T = TypeVar("T", bound=BaseModel)


class Gemini(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)
        self.model_name = DEFAULT_MODEL

    @cached_property
    def client(self, model_name: str = DEFAULT_MODEL):
        """The raw Gemini client."""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        self.model_name = model_name
        return genai.GenerativeModel(model_name=self.model_name)

    @cached_property
    def structured_client(self):
        """A Gemini client patched with Instructor."""
        return instructor.from_gemini(self.client)

    def send_conversation(self, conversation: "Conversation") -> "Message":
        """Send a conversation to the Gemini API."""
        from ..models import Message

        messages = [
            {
                "role": msg.role,
                "content": msg.text,
                "metadata": msg.meta or {},
            }
            for msg in conversation.messages
        ]

        try:
            response = self.structured_client.chat.completions.create(
                messages=messages, response_model=None
            )
        except Exception as e:
            # Handle the exception appropriately, e.g., log the error or raise a custom exception
            raise RuntimeError(f"Failed to send conversation to Gemini API: {e}") from e

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=str(response),
            raw=response,
            llm_model=self.model_name,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, prompt: str, response_model: Type[T], **kwargs) -> T:
        """Send a structured response to the Gemini API."""
        try:
            response = self.structured_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                **kwargs,
            )
        except Exception as e:
            # Handle the exception appropriately, e.g., log the error or raise a custom exception
            raise RuntimeError(
                f"Failed to send structured response to Gemini API: {e}"
            ) from e
        return response

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemini API."""
        try:
            response = self.structured_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=None,
                **kwargs,
            )
        except Exception as e:
            # Handle the exception appropriately, e.g., log the error or raise a custom exception
            raise RuntimeError(f"Failed to generate text with Gemini API: {e}") from e
        return str(response)
