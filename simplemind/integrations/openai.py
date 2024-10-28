import os

import instructor
from openai import OpenAI as BaseOpenAI

from .base import BaseClientProvider


class OpenAI(BaseClientProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.login()

    def login(self):
        """Initialize OpenAI client, with Instructor enabled."""

        # Default to environment variable if not provided.
        if self._api_key is None:
            self._api_key = os.getenv("OPENAI_API_KEY")

        base_client = BaseOpenAI(api_key=self._api_key)
        self.client = instructor.from_openai(base_client)
        self.test_connection()


    @property
    def available_models(self):
        """Returns the available models from the OpenAI client."""

        def gen():
            for model in self.client.models.list():
                yield model.id

        return [g for g in gen()]

    def test_connection(self):
       pass
