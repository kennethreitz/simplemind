from types import SimpleNamespace

import pytest

import simplemind as sm
from simplemind.providers import Gemini

genai = pytest.importorskip("google.generativeai")

CUSTOM_MODEL = "models/gemini-pro-latest"


class MockModel:
    """Stands in for `genai.GenerativeModel`, recording how it was used."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.sent = []

    def start_chat(self):
        return self

    def send_message(self, text):
        self.sent.append(text)
        return SimpleNamespace(text="bonjour")

    def generate_content(self, prompt, stream=False, **kwargs):
        self.sent.append(prompt)
        if stream:
            return [SimpleNamespace(text="bon"), SimpleNamespace(text="jour")]
        return SimpleNamespace(text="bonjour")


@pytest.fixture
def created_models(monkeypatch):
    """Mock the Gemini SDK, so no API key or network is needed.

    Returns the list of `GenerativeModel` instances the provider creates.
    """
    created = []

    def generative_model(model_name):
        model = MockModel(model_name)
        created.append(model)
        return model

    monkeypatch.setattr(genai, "GenerativeModel", generative_model)
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)

    provider = Gemini(api_key="test-key")
    monkeypatch.setattr("simplemind.models.find_provider", lambda name: provider)
    monkeypatch.setattr("simplemind.find_provider", lambda name: provider)
    return created


def test_send_conversation(created_models):
    conv = sm.create_conversation(llm_provider="gemini")
    conv.add_message("user", "Translate to French: hello")

    # Conversation.send() always forwards `tools`, which Gemini must accept.
    response = conv.send()

    assert response.text == "bonjour"
    assert response.llm_provider == "gemini"
    assert [m.model_name for m in created_models] == [Gemini.DEFAULT_MODEL]
    assert response.llm_model == Gemini.DEFAULT_MODEL


def test_send_conversation_uses_conversation_model(created_models):
    conv = sm.create_conversation(llm_provider="gemini", llm_model=CUSTOM_MODEL)
    conv.add_message("user", "Translate to French: hello")

    response = conv.send()

    assert [m.model_name for m in created_models] == [CUSTOM_MODEL]
    assert response.llm_model == CUSTOM_MODEL


def test_send_conversation_with_tools_is_not_supported(created_models):
    def get_location():
        """Get the current location"""
        return "San Francisco, CA"

    conv = sm.create_conversation(llm_provider="gemini")
    conv.add_message("user", "Where am I?")

    with pytest.raises(NotImplementedError, match="Gemini"):
        conv.send(tools=[get_location])


def test_generate_text_uses_default_model(created_models):
    text = sm.generate_text("Say hello", llm_provider="gemini")

    assert text == "bonjour"
    assert [m.model_name for m in created_models] == [Gemini.DEFAULT_MODEL]


def test_generate_text_uses_requested_model(created_models):
    sm.generate_text("Say hello", llm_provider="gemini", llm_model=CUSTOM_MODEL)

    assert [m.model_name for m in created_models] == [CUSTOM_MODEL]


def test_generate_text_without_llm_model_argument(created_models):
    # Calling the provider directly, without `llm_model`, must fall back to the default.
    assert Gemini(api_key="test-key").generate_text("Say hello") == "bonjour"
    assert [m.model_name for m in created_models] == [Gemini.DEFAULT_MODEL]


def test_generate_stream_text_uses_requested_model(created_models):
    chunks = list(
        sm.generate_text(
            "Say hello", llm_provider="gemini", llm_model=CUSTOM_MODEL, stream=True
        )
    )

    assert "".join(chunks) == "bonjour"
    assert [m.model_name for m in created_models] == [CUSTOM_MODEL]


def test_structured_response_uses_requested_model(created_models, monkeypatch):
    class Greeting(sm.BaseModel):
        text: str

    patched_with = []

    def from_gemini(client):
        patched_with.append(client.model_name)
        create = lambda **kwargs: Greeting(text="bonjour")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("simplemind.providers.gemini.instructor.from_gemini", from_gemini)

    result = sm.generate_data(
        "Say hello", response_model=Greeting, llm_provider="gemini", llm_model=CUSTOM_MODEL
    )

    assert result.text == "bonjour"
    assert patched_with == [CUSTOM_MODEL]


def test_clients_are_reused_per_model(created_models):
    for model in (None, None, CUSTOM_MODEL, CUSTOM_MODEL):
        sm.generate_text("Say hello", llm_provider="gemini", llm_model=model)

    assert [m.model_name for m in created_models] == [Gemini.DEFAULT_MODEL, CUSTOM_MODEL]
