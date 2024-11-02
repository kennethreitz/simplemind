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


PROVIDER_NAME = "xai"
DEFAULT_MODEL = "grok-beta"
BASE_URL = "https://api.x.ai/v1"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_KWARGS = {"max_tokens": DEFAULT_MAX_TOKENS}


class XAI(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_KWARGS = DEFAULT_KWARGS
    supports_streaming = True
    supports_structured_responses = False

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)

    @cached_property
    def client(self):
        """The raw OpenAI client."""
        if not self.api_key:
            raise ValueError("XAI API key is required")
        try:
            import openai as oa
        except ImportError as exc:
            raise ImportError(
                "Please install the `openai` package: `pip install openai`"
            ) from exc
        return oa.OpenAI(
            api_key=self.api_key,
            base_url=BASE_URL,
        )

    @cached_property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_openai(self.client)

    @logger
    def send_conversation(self, conversation: "Conversation", **kwargs) -> "Message":
        """Send a conversation to the OpenAI API."""
        from ..models import Message

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.chat.completions.create(
            model=conversation.llm_model or self.DEFAULT_MODEL,
            messages=messages,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        # Get the response content from the OpenAI response
        assistant_message = response.choices[0].message

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message.content,
            raw=response,
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    @logger
    def structured_response(
        self, prompt: str, response_model: Type[T], *, llm_model: str
    ) -> T:
        raise NotImplementedError("XAI does not support structured responses")

    @logger
    def generate_text(self, prompt: str, *, llm_model: str, **kwargs) -> str:
        # Prepare the messages.
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Make the request.
        response = self.client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        # Return the response content.
        return str(response.choices[0].message.content)

    @logger
    def generate_stream_text(
        self, prompt: str, *, llm_model: str, **kwargs
    ) -> Iterator[str]:
        # Prepare the messages.
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Make the request.
        response = self.client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            stream=True,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        # Iterate over the response and yield the content.
        for chunk in response:
            yield chunk.choices[0].delta.content
