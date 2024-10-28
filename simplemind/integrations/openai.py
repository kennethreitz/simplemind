import os

import instructor
from openai import OpenAI

from .base import BaseClientProvider


class OpenAI(BaseClientProvider):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
        self.login()



    def login(self):
        """Initialize OpenAI client, with Instructor enabled."""

        # Default to environment variable if not provided.
        if self._api_key is None:
            self._api_key = os.getenv("OPENAI_API_KEY")

        base_client = OpenAI(api_key=self._api_key)
        self.client = instructor.from_openai(base_client)
        self.test_connection()


    def available_models(self):
        pass

    def test_connection(self):
        try:
            # openai.api_key = self._api_key
            self.client.models.list()
            # self.logger.info("OpenAI connection test successful")
        except Exception as e:
            # self.logger.error(f"OpenAI connection test failed: {str(e)}")
            raise
