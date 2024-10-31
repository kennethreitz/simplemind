from typing import Union
import json

import instructor
import anthropic

from ._base import BaseProvider
from ..settings import settings

PROVIDER_NAME = "amazon"
DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_MAX_TOKENS = 5000

class Amazon(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: Union[str, None] = None):
        self.api_key = api_key or settings.get_api_key(PROVIDER_NAME)

    @property
    def client(self):
        """The AnthropicBedrock client."""
        if not self.api_key:
            raise ValueError("Profile name is not provided")

        return anthropic.AnthropicBedrock(aws_profile=self.api_key)

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_anthropic(self.client)

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

    def structured_response(self, prompt, response_model, *, llm_model: str, **kwargs):
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.structured_client.chat.completions.create(
            messages=messages,
            model=llm_model or self.DEFAULT_MODEL,
            response_model=response_model,
            max_tokens=DEFAULT_MAX_TOKENS,
            **kwargs,
        )
        return response

    def generate_text(self, prompt, *, llm_model, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model or self.DEFAULT_MODEL,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **kwargs,
        )

        return response.content[0].text