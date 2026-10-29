from types import SimpleNamespace

import pytest

import simplemind as sm
from simplemind.providers import Gemini


class MockChat:
    def __init__(self):
        self.sent = []

    def send_message(self, text):
        self.sent.append(text)
        return SimpleNamespace(text="bonjour")


@pytest.fixture
def gemini(monkeypatch):
    """A Gemini provider whose client is mocked, so no API key or network is needed."""
    provider = Gemini(api_key="test-key")
    chat = MockChat()
    provider.__dict__["client"] = SimpleNamespace(start_chat=lambda: chat)
    monkeypatch.setattr("simplemind.models.find_provider", lambda name: provider)
    return provider


def test_send_conversation(gemini):
    conv = sm.create_conversation(llm_provider="gemini")
    conv.add_message("user", "Translate to French: hello")

    # Conversation.send() always forwards `tools`, which Gemini must accept.
    response = conv.send()

    assert response.text == "bonjour"
    assert response.llm_provider == "gemini"


def test_send_conversation_with_tools_is_not_supported(gemini):
    def get_location():
        """Get the current location"""
        return "San Francisco, CA"

    conv = sm.create_conversation(llm_provider="gemini")
    conv.add_message("user", "Where am I?")

    with pytest.raises(NotImplementedError, match="Gemini"):
        conv.send(tools=[get_location])
