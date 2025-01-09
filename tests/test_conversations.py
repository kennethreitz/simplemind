import json

import pytest

import simplemind as sm
from simplemind.models import BasePlugin, Conversation
from simplemind.providers import Anthropic, Gemini, Groq, Ollama, OpenAI


@pytest.mark.parametrize(
    "provider_cls",
    [
        Anthropic,
        Gemini,
        OpenAI,
        Groq,
        Ollama,
        # Amazon
    ],
)
def test_generate_data(provider_cls):
    conv = sm.create_conversation(
        llm_model=provider_cls.DEFAULT_MODEL, llm_provider=provider_cls.NAME
    )

    conv.add_message(text="hey")
    data = conv.send()

    assert isinstance(data.text, str)
    assert len(data.text) > 0


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    conv = Conversation(llm_provider="openai")
    conv.add_message(role="user", text="Hello!")
    conv.add_message(role="assistant", text="Hi there!")
    conv.add_message(role="user", text="How are you?")
    return conv


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary file path for testing."""
    return tmp_path / "conversation.json"


def test_save_conversation(sample_conversation, temp_json_file):
    """Test saving a conversation to a JSON file."""
    sample_conversation.save(temp_json_file)

    assert temp_json_file.exists()

    with open(temp_json_file) as f:
        saved_data = json.load(f)

    assert "id" in saved_data
    assert "messages" in saved_data
    assert "llm_model" in saved_data
    assert "llm_provider" in saved_data

    assert len(saved_data["messages"]) == 3
    assert saved_data["messages"][0]["text"] == "Hello!"
    assert saved_data["messages"][1]["text"] == "Hi there!"
    assert saved_data["messages"][2]["text"] == "How are you?"


def test_load_conversation(sample_conversation, temp_json_file):
    """Test loading a conversation from a JSON file."""
    sample_conversation.save(temp_json_file)

    loaded_conv = Conversation.load(temp_json_file)

    assert loaded_conv.id == sample_conversation.id
    assert loaded_conv.llm_model == sample_conversation.llm_model
    assert loaded_conv.llm_provider == sample_conversation.llm_provider
    assert len(loaded_conv.messages) == len(sample_conversation.messages)

    for original_msg, loaded_msg in zip(
        sample_conversation.messages, loaded_conv.messages
    ):
        assert loaded_msg.role == original_msg.role
        assert loaded_msg.text == original_msg.text
        assert loaded_msg.meta == original_msg.meta


def test_save_load_with_plugins(sample_conversation, temp_json_file):
    """Test that plugins are properly excluded from serialization."""

    # Create a dummy plugin
    class DummyPlugin(BasePlugin):
        def initialize_hook(self, conversation):
            pass

    sample_conversation.add_plugin(DummyPlugin())

    sample_conversation.save(temp_json_file)
    loaded_conv = Conversation.load(temp_json_file)

    assert len(loaded_conv.plugins) == 0
