import os
from typing import List, Optional

import instructor
from anthropic import Anthropic as BaseAnthropic

from .base import BaseClientProvider
from ..models import AIResponse, Conversation
from ..logger import logger


class Anthropic(BaseClientProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)
        self.login()

    def login(self):
        if not self._api_key:
            self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key not provided.")
        base_client = BaseAnthropic(api_key=self._api_key)
        self.client = instructor.from_anthropic(base_client)
        if not self.test_connection():
            raise ConnectionError("Failed to connect to Anthropic API.")
        logger.info("Logged in to Anthropic successfully.")

    @property
    def available_models(self) -> List[str]:
        try:
            return [
                "claude-3-opus-20240229",
                "claude-3-5-sonnet-20240620",
                "claude-3-haiku-20240307",
            ]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    def test_connection(self) -> bool:
        models = self.available_models
        if models:
            logger.info(f"Available models: {models}")
            return True
        logger.warning("No available models found.")
        return False

    def generate_response(self, conversation: Conversation) -> AIResponse:
        messages = [
            {"role": msg.role, "content": msg.content} for msg in conversation.messages
        ]
        params = {
            "messages": messages,
            "model": self.model,
        }
        if conversation.context:
            params["context"] = conversation.context

        try:
            completion = self.client.completions.create(**params)
            response_text = completion.completion
            metadata = {"model": completion.model, "usage": completion.usage}
            logger.info("Generated response from Anthropic.")
            return AIResponse(
                text=response_text, response=completion, metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e
