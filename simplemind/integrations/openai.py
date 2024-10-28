import os

import instructor
from openai import OpenAI as BaseOpenAI

from .base import BaseClientProvider
from ..models import AIResponse


class OpenAI(BaseClientProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.login()

    def login(self):
        """Initialize OpenAI client, with Instructor enabled."""

        # Default to environment variable if not provided.
        if self._api_key is None:
            self._api_key = os.getenv("OPENAI_API_KEY")

        self.client = BaseOpenAI(api_key=self._api_key)
        self.instructor_client = instructor.from_openai(self.client)
        assert self.test_connection()

    @property
    def available_models(self):
        """Returns the available models from the OpenAI client."""

        def gen():
            for model in self.client.models.list():
                yield model.id

        return [g for g in gen()]

    def test_connection(self):
        """Test the connection to OpenAI. Returns True if successful."""

        return bool(len(self.available_models))

    def message(self, message, *, response_model=False, **kwargs):
        """Generates a response from the OpenAI client."""
        use_instructor = bool(response_model)

        client = self.instructor_client if use_instructor else self.client

        # Parameters for the OpenAI client.
        params = {
            "messages": [{"role": "user", "content": message}],
            "model": self.model,
        }
        params.update(kwargs)

        if use_instructor:
            params["response_model"] = response_model

        # Make the request to OpenAI.
        completion = client.chat.completions.create(**params)

        if use_instructor:
            return completion.model_dump()

        else:
            return AIResponse(
                response=completion,
                text=completion.choices[0].message.content,
            )
