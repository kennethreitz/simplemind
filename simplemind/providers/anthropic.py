from typing import Union

import anthropic
import instructor

from simplemind.models import BaseProvider, Conversation, Message
from simplemind.settings import settings

PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1000


class Anthropic(BaseProvider):
    __name__ = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: Union[str, None] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY

    @property
    def client(self):
        """The raw Anthropic client."""
        return anthropic.Anthropic(api_key=self.api_key)

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_anthropic(self.client)

    def send_conversation(self, conversation: "Conversation"):
        """Send a conversation to the Anthropic API."""
        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.messages.create(
            model=conversation.llm_model or DEFAULT_MODEL,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # Get the response content from the OpenAI response
        # assistant_message = response.choices[0].message
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

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.messages.create(
            model=llm_model, messages=messages, max_tokens=DEFAULT_MAX_TOKENS
        )

        return response.content[0].text
