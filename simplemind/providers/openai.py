from functools import cached_property
from typing import Type, TypeVar

import instructor
import openai as oa
from pydantic import BaseModel

from ..settings import settings
from ._base import BaseProvider

T = TypeVar("T", bound=BaseModel)

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAI(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)

    @cached_property
    def client(self):
        """The raw OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        return oa.OpenAI(api_key=self.api_key)

    @cached_property
    def structured_client(self):
        """A OpenAI client with Instructor."""
        return instructor.from_openai(self.client)

    def send_conversation(self, conversation: "Conversation", **kwargs):
        """Send a conversation to the OpenAI API."""
        from ..models import Message

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.chat.completions.create(
            model=conversation.llm_model or DEFAULT_MODEL, messages=messages, **kwargs
        )

        # Get the response content from the OpenAI response
        assistant_message = response.choices[0].message

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message.content or "",
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        llm_model: str | None = None,
        **kwargs,
    ) -> T:
        """Get a structured response from the OpenAI API."""
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.structured_client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            response_model=response_model,
            **kwargs,
        )
        return response

    def generate_text(self, prompt: str, *, llm_model: str | None = None, **kwargs):
        """Generate text using the OpenAI API."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            messages=messages, model=llm_model or self.DEFAULT_MODEL, **kwargs
        )
        return response.choices[0].message.content
