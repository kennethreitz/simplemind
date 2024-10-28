import unittest
from unittest.mock import patch, MagicMock
from simplemind.providers.openai import OpenAI
from simplemind.core.errors import AuthenticationError, ProviderError


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"

    @patch("simplemind.integrations.openai.BaseOpenAI")
    def test_initialization(self, mock_openai):
        provider = OpenAI(api_key=self.api_key)
        self.assertIsNotNone(provider.client)

    def test_missing_api_key(self):
        with self.assertRaises(AuthenticationError):
            OpenAI(api_key=None)


if __name__ == "__main__":
    unittest.main()
