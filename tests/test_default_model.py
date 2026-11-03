from unittest import mock

import pytest

import simplemind as sm
from simplemind.settings import Settings


@pytest.fixture
def provider():
    """A fake provider, so no API key or network is needed."""
    fake = mock.MagicMock()
    with mock.patch("simplemind.find_provider", return_value=fake):
        yield fake


def test_default_model_is_read_from_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
    assert Settings().DEFAULT_LLM_MODEL == "gpt-4.1-mini"


def test_empty_default_model_env_is_none(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "")
    assert Settings().DEFAULT_LLM_MODEL is None


def test_conversation_uses_default_model(monkeypatch):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    assert sm.create_conversation().llm_model == "env-model"


def test_explicit_model_overrides_default_in_conversation(monkeypatch):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    conv = sm.create_conversation(llm_model="explicit-model")
    assert conv.llm_model == "explicit-model"


def test_generate_text_uses_default_model(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    sm.generate_text("hi")
    assert provider.generate_text.call_args.kwargs["llm_model"] == "env-model"


def test_generate_text_explicit_model_wins(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    sm.generate_text("hi", llm_model="explicit-model")
    assert provider.generate_text.call_args.kwargs["llm_model"] == "explicit-model"


def test_generate_text_stream_uses_default_model(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    provider.supports_streaming = True
    sm.generate_text("hi", stream=True)
    assert provider.generate_stream_text.call_args.kwargs["llm_model"] == "env-model"


def test_generate_data_uses_default_model(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    sm.generate_data("hi", response_model=object)
    assert provider.structured_response.call_args.kwargs["llm_model"] == "env-model"


def test_session_uses_default_model(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", "env-model")
    sm.Session(llm_provider="openai").generate_text("hi")
    assert provider.generate_text.call_args.kwargs["llm_model"] == "env-model"


def test_no_default_passes_none(monkeypatch, provider):
    monkeypatch.setattr(sm.settings, "DEFAULT_LLM_MODEL", None)
    sm.generate_text("hi")
    assert provider.generate_text.call_args.kwargs["llm_model"] is None
