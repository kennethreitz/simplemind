import os
from typing import Optional, List
import instructor
from openai import OpenAI as BaseOpenAI

from .base import BaseClientProvider
from ..models import AIResponse, Conversation
from ..logger import logger
from simplemind.config import settings

class OpenAI(BaseClientProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)
        self.login()

    def login(self):
        if not self._api_key:
            self._api_key = settings.openai_api_key
        if not self._api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = BaseOpenAI(api_key=self._api_key)
        self.instructor_client = instructor.from_openai(self.client)
        if not self.test_connection():
            raise ConnectionError("Failed to connect to OpenAI API.")
        logger.info("Logged in to OpenAI successfully.")

    @property
    def available_models(self) -> List[str]:
        try:
            return [model.id for model in self.client.models.list()]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    def test_connection(self) -> bool:
        try:
            models = self.available_models
            if models:
                logger.info(f"Available models: {models}")
                return True
            else:
                logger.warning("No available models found.")
                return False
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
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
            completion = self.client.chat.completions.create(**params)
            response_text = completion.choices[0].message.content
            metadata = {"model": completion.model, "usage": completion.usage}
            logger.info("Generated response from OpenAI.")
            return AIResponse(text=response_text, response=completion, metadata=metadata)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e
