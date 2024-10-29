from typing import Union

import instructor
import openai as oa

from simplemind.models import BaseProvider, Conversation, Message
from simplemind.settings import settings

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAI(BaseProvider):
    __name__ = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: Union[str, None] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY

    @property
    def client(self):
        """The raw OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        return oa.OpenAI(api_key=self.api_key)

    @property
    def structured_client(self):
        """A OpenAI client with Instructor."""
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
            text=assistant_message.content or "",
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, prompt, response_model, *, llm_model):
        # Ensure messages are provided in kwargs
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.structured_client.chat.completions.create(
            messages=messages, model=llm_model, response_model=response_model
        )
        return response

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            messages=messages, model=llm_model
        )

        return response.choices[0].message.content
