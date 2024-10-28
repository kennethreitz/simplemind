import unittest
from unittest.mock import patch, MagicMock
from simplemind.providers.openai import OpenAI
from simplemind.core.errors import AuthenticationError, ProviderError


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"

    @patch("simplemind.providers.openai.openai")
    def test_initialization(self, mock_openai):
        mock_openai.Model.list.return_value = {"data": ["gpt-4"]}
        provider = OpenAI(api_key=self.api_key)
        self.assertIsNotNone(provider.client)
        mock_openai.Model.list.assert_called_once()

    def test_missing_api_key(self):
        with self.assertRaises(AuthenticationError):
            OpenAI(api_key=None)


if __name__ == "__main__":
    unittest.main()
