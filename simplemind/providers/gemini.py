# TODO: this is a placeholder file for the Gemini provider
# IT is not currently working as desired.

from typing import TYPE_CHECKING, Callable, Iterator, Type, TypeVar

import instructor
from pydantic import BaseModel

from ..logging import logger
from ..settings import settings
from ._base import BaseProvider
from ._base_tools import BaseTool

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class Gemini(BaseProvider):
    NAME = "gemini"
    DEFAULT_MODEL = "models/gemini-3.8-flash"
    supports_streaming = True

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(self.NAME)
        self.model_name = self.DEFAULT_MODEL
        self._clients = {}

    def set_model(self, model_name: str):
        self.model_name = model_name

    def _get_client(self, model_name: str | None = None):
        """The raw Gemini client for a model, or for the default model if omitted."""
        model_name = model_name or self.model_name
        if model_name not in self._clients:
            if not self.api_key:
                raise ValueError("Gemini API key is required")
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ImportError(
                    "Please install the `google-generativeai` package: `pip install google-generativeai`"
                ) from exc
            genai.configure(api_key=self.api_key)
            self._clients[model_name] = genai.GenerativeModel(model_name=model_name)
        return self._clients[model_name]

    @property
    def client(self):
        """The raw Gemini client for the default model."""
        return self._get_client()

    @property
    def structured_client(self):
        """A Gemini client for the default model, patched with Instructor."""
        return instructor.from_gemini(self.client)

    @logger
    def send_conversation(
        self,
        conversation: "Conversation",
        tools: list[Callable | BaseTool] | None = None,
    ) -> "Message":
        """Send a conversation to the Gemini API."""
        from ..models import Message

        if tools:
            raise NotImplementedError("Tool calling is not yet supported for Gemini.")

        model_name = conversation.llm_model or self.model_name

        # Convert messages to Gemini's format
        chat = self._get_client(model_name).start_chat()

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
            llm_model=model_name,
            llm_provider=self.NAME,
        )

    @logger
    def structured_response(self, prompt: str, response_model: Type[T], **kwargs) -> T:
        """Send a structured response to the Gemini API."""
        llm_model = kwargs.pop("llm_model", None)

        try:
            structured_client = instructor.from_gemini(self._get_client(llm_model))
            response = structured_client.chat.completions.create(
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
        llm_model = kwargs.pop("llm_model", None)
        try:
            response = self._get_client(llm_model).generate_content(prompt, **kwargs)
        except Exception as e:
            # Handle the exception appropriately, e.g., log the error or raise a custom exception
            raise RuntimeError(f"Failed to generate text with Gemini API: {e}") from e
        return response.text

    @logger
    def generate_stream_text(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming text using the Gemini API."""
        llm_model = kwargs.pop("llm_model", None)
        try:
            response = self._get_client(llm_model).generate_content(
                prompt, stream=True, **kwargs
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate streaming text with Gemini API: {e}"
            ) from e
