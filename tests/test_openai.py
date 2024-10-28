import unittest
from unittest.mock import patch, MagicMock
from simplemind.integrations.openai import OpenAI


class TestOpenAIProvider(unittest.TestCase):
    @patch("simplemind.integrations.openai.BaseOpenAI")
    def setUp(self, mock_openai):
        self.mock_openai = mock_openai.return_value
        self.mock_openai.models.list.return_value = [MagicMock(id="gpt-4")]
        self.provider = OpenAI(api_key="test_api_key", model="gpt-4")

    def test_available_models(self):
        models = self.provider.available_models
        self.assertIn("gpt-4", models)

    def test_test_connection_success(self):
        self.assertTrue(self.provider.test_connection())

    def test_generate_response_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.provider.generate_response(None)


if __name__ == "__main__":
    unittest.main()
