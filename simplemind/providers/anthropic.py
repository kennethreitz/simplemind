from typing import Union

import anthropic
import instructor

from ._base import BaseProvider
from ..settings import settings

PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1000


class Anthropic(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: Union[str, None] = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)

    @property
    def client(self):
        """The raw Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        return anthropic.Anthropic(api_key=self.api_key)

    @property
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
            model=conversation.llm_model or DEFAULT_MODEL,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **kwargs,
        )

        # Get the response content from the Anthropic response
        assistant_message = response.content[0].text

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message,
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, model, response_model, **kwargs):
        response = self.structured_client.messages.create(
            model=model, response_model=response_model, **kwargs
        )
        return response

    def generate_text(self, prompt, *, llm_model, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **kwargs,
        )

        return response.content[0].text
