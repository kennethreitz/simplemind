import os

import instructor
from anthropic import Anthropic as BaseAnthropic

from .base import BaseClientProvider


class Anthropic(BaseClientProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.login()

    def login(self):
        """Initialize Anthropic client, with Instructor enabled."""

        # Default to environment variable if not provided.
        if self._api_key is None:
            self._api_key = os.getenv("ANTHROPIC_API_KEY")

        base_client = BaseAnthropic(api_key=self._api_key)
        self.client = instructor.from_anthropic(base_client)
        # assert self.test_connection()

    @property
    def available_models(self):
        """Returns the available models from the Anthropic client."""

        # TODO: scrape from website or embed
        return [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20240620",
        ]

    # def test_connection(self):
    #     """Test the connection to Anthropic. Returns True if successful."""

    #     raise NotImplementedError("Anthropic test_connection not implemented.")
