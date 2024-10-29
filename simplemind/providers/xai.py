from typing import Union

import instructor
import openai as oa

from simplemind.models import BaseProvider, Conversation, Message
from simplemind.settings import settings

PROVIDER_NAME = "xai"
DEFAULT_MODEL = "grok-beta"
BASE_URL = "https://api.x.ai/v1"
DEFAULT_MAX_TOKENS = 1000


class XAI(BaseProvider):
    __name__ = PROVIDER_NAME
    DEFAULT_MODEL: str = DEFAULT_MODEL

    def __init__(self, api_key: Union[str, None] = None):
        self.api_key = api_key or settings.XAI_API_KEY

    @property
    def client(self):
        """The raw OpenAI client."""

        return oa.OpenAI(
            api_key=settings.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_openai(self.client)

    def send_conversation(self, conversation: "Conversation"):
        """Send a conversation to the OpenAI API."""

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        response = self.client.chat.completions.create(
            model=conversation.llm_model or DEFAULT_MODEL, messages=messages
        )

        # Get the response content from the OpenAI response
        assistant_message = response.choices[0].message

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message.content,
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, prompt: str, response_model, *, llm_model):
        raise NotImplementedError("XAI does not support structured responses")

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            messages=messages, model=llm_model
        )

        return response.choices[0].message.content
