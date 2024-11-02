# TODO: this is a placeholder file for the Gemini provider
# IT is not currently working as desired.

from functools import cached_property
from typing import TYPE_CHECKING, Type, TypeVar, Iterator

import instructor
from pydantic import BaseModel

from ..logging import logger
from ..settings import settings
from ._base import BaseProvider

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


PROVIDER_NAME = "gemini"
DEFAULT_MODEL = "models/gemini-1.5-flash-latest"


class Gemini(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL
    supports_streaming = True

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)
        self.model_name = DEFAULT_MODEL

    def set_model(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def client(self):
        """The raw Gemini client."""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Please install the `google-generativeai` package: `pip install google-generativeai`"
            ) from exc
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(model_name=self.model_name)

    @cached_property
    def structured_client(self):
        """A Gemini client patched with Instructor."""
        return instructor.from_gemini(self.client)

    @logger
    def send_conversation(self, conversation: "Conversation") -> "Message":
        """Send a conversation to the Gemini API."""
        from ..models import Message

        # Convert messages to Gemini's format
        chat = self.client.start_chat()

        # Send all previous messages to establish context
        for msg in conversation.messages[:-1]:  # All messages except the last one
            chat.send_message(msg.text)

        # Send the final message and get response
        try:
            response = chat.send_message(conversation.messages[-1].text)
        except Exception as e:
            raise RuntimeError(f"Failed to send conversation to Gemini API: {e}") from e

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=response.text,
            raw=response,
            llm_model=self.model_name,
            llm_provider=PROVIDER_NAME,
        )

    @logger
    def structured_response(self, prompt: str, response_model: Type[T], **kwargs) -> T:
        """Send a structured response to the Gemini API."""
        # Only try to pop if the key exists
        kwargs.pop("llm_model", None)  # Add default value of None

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
        return response_model.model_validate(response)

    @logger
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemini API."""
        kwargs.pop("llm_model")
        try:
            response = self.client.generate_content(prompt, **kwargs)
        except Exception as e:
            # Handle the exception appropriately, e.g., log the error or raise a custom exception
            raise RuntimeError(f"Failed to generate text with Gemini API: {e}") from e
        return response.text

    @logger
    def generate_stream_text(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming text using the Gemini API."""
        kwargs.pop("llm_model", None)
        try:
            response = self.client.generate_content(prompt, stream=True, **kwargs)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate streaming text with Gemini API: {e}"
            ) from e
