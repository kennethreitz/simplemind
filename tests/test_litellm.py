"""Offline tests for the LiteLLM provider."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from simplemind.models import Conversation
from simplemind.providers import LiteLLM, providers
from simplemind.settings import settings
from simplemind.utils import find_provider

VIDEO_URL = "https://example.invalid/clip.mp4"


class _Answer(BaseModel):
    answer: str


def _response(content="ok", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name, arguments, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=function)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Stand in for the `litellm` package so no request leaves the process."""
    module = types.ModuleType("litellm")
    module.completion = MagicMock(return_value=_response())
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


@pytest.fixture
def provider(fake_litellm, monkeypatch):
    monkeypatch.setattr(settings, "LITELLM_API_KEY", None)
    monkeypatch.setattr(settings, "LITELLM_API_BASE", None)
    return LiteLLM()


def _sent(fake_litellm):
    return fake_litellm.completion.call_args.kwargs


def test_litellm_registered():
    assert LiteLLM in providers
    assert isinstance(find_provider("litellm"), LiteLLM)
    assert isinstance(find_provider("LiteLLM"), LiteLLM)


def test_generate_text_without_credentials(provider, fake_litellm):
    result = provider.generate_text(
        prompt="What is 2+2?", llm_model="anthropic/claude-sonnet-4-5"
    )

    assert result == "ok"
    sent = _sent(fake_litellm)
    assert sent["model"] == "anthropic/claude-sonnet-4-5"
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}
    ]
    assert sent["drop_params"] is True
    # Unset, LiteLLM reads the upstream provider's own credentials from the environment.
    assert "api_key" not in sent
    assert "api_base" not in sent
    assert fake_litellm.suppress_debug_info is True


def test_generate_text_defaults_model(provider, fake_litellm):
    provider.generate_text(prompt="hi")

    assert _sent(fake_litellm)["model"] == LiteLLM.DEFAULT_MODEL


def test_credentials_sent_when_set(fake_litellm, monkeypatch):
    monkeypatch.setattr(settings, "LITELLM_API_KEY", None)
    monkeypatch.setattr(settings, "LITELLM_API_BASE", "http://localhost:4000/")

    LiteLLM(api_key="sk-proxy").generate_text(
        prompt="hi", llm_model="litellm_proxy/my-alias"
    )

    sent = _sent(fake_litellm)
    assert sent["api_key"] == "sk-proxy"
    assert sent["api_base"] == "http://localhost:4000"


def test_credentials_read_from_settings(fake_litellm, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
    monkeypatch.setattr(
        settings, "LITELLM_API_KEY", type(settings)().LITELLM_API_KEY
    )
    monkeypatch.setattr(settings, "LITELLM_API_BASE", None)

    LiteLLM().generate_text(prompt="hi")

    assert _sent(fake_litellm)["api_key"] == "sk-from-env"


def test_call_kwargs_override_defaults(provider, fake_litellm):
    provider.generate_text(prompt="hi", drop_params=False, temperature=0.2)

    sent = _sent(fake_litellm)
    assert sent["drop_params"] is False
    assert sent["temperature"] == 0.2


@pytest.mark.parametrize(
    ("kwargs", "expected_types"),
    [
        ({}, ["text"]),
        ({"image_url": "data:image/png;base64,AAAA"}, ["text", "image_url"]),
        ({"video_url": VIDEO_URL}, ["text", "video_url"]),
    ],
)
def test_generate_text_media_parts(provider, fake_litellm, kwargs, expected_types):
    provider.generate_text(prompt="describe this", **kwargs)

    content = _sent(fake_litellm)["messages"][0]["content"]
    assert [part["type"] for part in content] == expected_types


def test_generate_stream_text(provider, fake_litellm):
    def chunk(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    fake_litellm.completion.return_value = iter(
        [chunk("Hel"), chunk(None), SimpleNamespace(choices=[]), chunk("lo")]
    )

    chunks = list(provider.generate_stream_text(prompt="hi", video_url=VIDEO_URL))

    assert chunks == ["Hel", "lo"]
    sent = _sent(fake_litellm)
    assert sent["stream"] is True
    assert sent["drop_params"] is True
    assert sent["messages"][0]["content"][-1]["type"] == "video_url"


def test_streaming_through_generate_text(provider, fake_litellm, monkeypatch):
    import simplemind as sm

    monkeypatch.setattr(sm, "find_provider", lambda name: provider)
    fake_litellm.completion.return_value = iter(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])]
    )

    assert list(sm.generate_text("hi", llm_provider="litellm", stream=True)) == ["ok"]


def test_structured_response(provider):
    structured = MagicMock()
    structured.chat.completions.create.return_value = {"answer": "4"}
    provider.__dict__["structured_client"] = structured

    response = provider.structured_response(
        prompt="What is 2+2?",
        response_model=_Answer,
        llm_model="gemini/gemini-2.5-flash",
    )

    assert response == _Answer(answer="4")
    sent = structured.chat.completions.create.call_args.kwargs
    assert sent["response_model"] is _Answer
    assert sent["model"] == "gemini/gemini-2.5-flash"
    assert sent["drop_params"] is True
    assert sent["messages"][0]["content"][0]["text"] == "What is 2+2?"


def test_structured_client_wraps_litellm_completion(provider, fake_litellm, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "simplemind.providers.litellm.instructor.from_litellm",
        lambda completion: seen.setdefault("completion", completion),
    )

    assert provider.structured_client is fake_litellm.completion
    assert seen["completion"] is fake_litellm.completion


def test_send_conversation(provider, fake_litellm):
    conversation = Conversation(llm_model="groq/llama-3.3-70b-versatile")
    conversation.add_message("system", "Be brief.")
    conversation.add_message("user", "Hi")

    message = provider.send_conversation(conversation)

    assert message.text == "ok"
    assert message.llm_provider == "litellm"
    assert message.llm_model == "groq/llama-3.3-70b-versatile"
    sent = _sent(fake_litellm)
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert "tools" not in sent


def test_send_conversation_runs_tools(provider, fake_litellm):
    def get_weather(city: str = Field(description="The city")) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    fake_litellm.completion.side_effect = [
        _response(content=None, tool_calls=[_tool_call("get_weather", '{"city": "Paris"}')]),
        _response(content="It is sunny in Paris."),
    ]
    conversation = Conversation(llm_model="anthropic/claude-sonnet-4-5")
    conversation.add_message("user", "Weather in Paris?")

    message = provider.send_conversation(conversation, tools=[get_weather])

    assert message.text == "It is sunny in Paris."
    first, second = fake_litellm.completion.call_args_list
    tools = first.kwargs["tools"]
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["parameters"]["required"] == ["city"]
    # The tool round-trip is appended to the messages sent on the follow-up call.
    assert second.kwargs["messages"][-2]["tool_calls"][0]["id"] == "call_1"
    assert second.kwargs["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sunny in Paris",
    }


def test_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(ImportError, match="simplemind\\[litellm\\]"):
        LiteLLM().generate_text(prompt="hi")
