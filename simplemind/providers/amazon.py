from functools import cached_property
from typing import TYPE_CHECKING, Iterator, Type, TypeVar

import instructor
from pydantic import BaseModel

from ..settings import settings
from ._base import BaseProvider

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class Amazon(BaseProvider):
    NAME = "amazon"
    DEFAULT_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_MAX_TOKENS = 5_000
    supports_streaming = True

    def __init__(self, profile_name: str | None = None):
        self.profile_name = profile_name or settings.AMAZON_PROFILE_NAME

    @cached_property
    def client(self):
        """The AnthropicBedrock client."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Please install the `anthropic` package: `pip install anthropic`"
            ) from exc

        if not self.profile_name:
            raise ValueError("Profile name is not provided")

        return anthropic.AnthropicBedrock(aws_profile=self.profile_name)

    @cached_property
    def structured_client(self) -> instructor.Instructor:
        """A client patched with Instructor."""

        return instructor.from_anthropic(self.client)

    def send_conversation(self, conversation: "Conversation", **kwargs) -> "Message":
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
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(
        self, prompt, response_model: Type[T], *, llm_model: str | None = None, **kwargs
    ) -> T:
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.structured_client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            response_model=response_model,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            **kwargs,
        )
        return response

    def generate_text(self, prompt: str, *, llm_model: str, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            **kwargs,
        )

        return response.content[0].text

    def generate_stream_text(
        self, prompt: str, *, llm_model: str, **kwargs
    ) -> Iterator[str]:
        """Generate streaming text using the Amazon API."""

        # Prepare the messages.
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Send the request to the API.
        response = self.client.messages.create(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            stream=True,
            **kwargs,
        )

        # Yield the text chunks.
        for chunk in response:
            if chunk.text:
                yield chunk.text
