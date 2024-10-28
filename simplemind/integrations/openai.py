from .base import BaseClientProvider


class OpenAI(BaseClientProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize()

    def initialize(self):
        assert self._api_key, "API key is required for OpenAI client"
        assert self._api_key.startswith("sk-"), "OpenAI API key must start with 'sk-'"

        self.logger.info("Initializing OpenAI client")
        self.logger.debug(f"API key: {self._api_key}")

        self.test_connection()



    def test_connection(self):
