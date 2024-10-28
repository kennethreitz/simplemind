from .base import BaseClientProvider


class OpenAI(BaseClientProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_connection(self):
