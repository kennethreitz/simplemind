from functools import cached_property
from typing import Type, TypeVar

import anthropic
import instructor
from pydantic import BaseModel

from ..settings import settings
from ._base import BaseProvider

T = TypeVar("T", bound=BaseModel)


PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1_000
DEFAULT_KWARGS = {"max_tokens": DEFAULT_MAX_TOKENS}


class Anthropic(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_KWARGS = DEFAULT_KWARGS

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)

    @cached_property
    def client(self):
        """The raw Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        return anthropic.Anthropic(api_key=self.api_key)

    @cached_property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_anthropic(self.client)

    def send_conversation(self, conversation: "Conversation", **kwargs):
        """Send a conversation to the Anthropic API."""
        from ..models import Message

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.messages.create(
            model=conversation.llm_model or self.DEFAULT_MODEL,
            messages=messages,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        # Get the response content from the Anthropic response
        assistant_message = response.content[0].text

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message,
            raw=response,
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(
        self, response_model: Type[T], *, llm_model: str | None = None, **kwargs
    ) -> T:
        model = llm_model or self.DEFAULT_MODEL

        # Extract the prompt from kwargs if it exists
        prompt = kwargs.pop("prompt", kwargs.pop("messages", ""))

        # Format the messages properly
        messages = [{"role": "user", "content": prompt}]

        response = self.structured_client.messages.create(
            model=model,
            messages=messages,  # Add the messages parameter
            response_model=response_model,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )
        return response

    def generate_text(self, prompt: str, *, llm_model: str, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            **{**self.DEFAULT_KWARGS, **kwargs},
        )

        return response.content[0].text
