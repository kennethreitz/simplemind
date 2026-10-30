"""Offline tests for the media parts of the chat-completions input path."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from simplemind.providers import OpenAI

IMAGE_URL = "https://example.invalid/frame.png"
VIDEO_URL = "https://example.invalid/clip.mp4"


class _Answer(BaseModel):
    answer: str


class _RecordingCompletions:
    """Records the request kwargs instead of calling a remote API."""

    def __init__(self, result):
        self.captured = None
        self._result = result

    def create(self, **kwargs):
        self.captured = kwargs
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
                    )
                ]
            )
        return self._result


class _RecordingClient:
    def __init__(self, result):
        self.completions = _RecordingCompletions(result)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def captured(self):
        return self.completions.captured


def _make_provider(structured=False):
    """Build a provider whose clients never reach the network."""
    provider = OpenAI(api_key="test-key")
    if structured:
        result = {"answer": "ok"}
    else:
        result = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))
            ]
        )
    client = _RecordingClient(result)
    # `client` and `structured_client` are cached properties, so seeding the
    # instance dict replaces them before any credentials are needed.
    provider.__dict__["client"] = client
    provider.__dict__["structured_client"] = client
    return provider, client


def _content(client):
    return client.captured["messages"][0]["content"]


def _part_types(client):
    return [part["type"] for part in _content(client)]


def test_generate_text_sends_text_only_by_default():
    provider, client = _make_provider()

    provider.generate_text(prompt="describe this", llm_model="test-model")

    assert _part_types(client) == ["text"]


@pytest.mark.parametrize(
    ("kwargs", "expected_types"),
    [
        ({"video_url": VIDEO_URL}, ["text", "video_url"]),
        ({"image_url": IMAGE_URL}, ["text", "image_url"]),
        (
            {"image_url": IMAGE_URL, "video_url": VIDEO_URL},
            ["text", "image_url", "video_url"],
        ),
    ],
)
def test_generate_text_sends_requested_media_parts(kwargs, expected_types):
    provider, client = _make_provider()

    provider.generate_text(prompt="describe this", llm_model="test-model", **kwargs)

    assert _part_types(client) == expected_types


def test_video_part_uses_the_documented_shape():
    provider, client = _make_provider()

    provider.generate_text(
        prompt="describe this", llm_model="test-model", video_url=VIDEO_URL
    )

    assert _content(client)[-1] == {
        "type": "video_url",
        "video_url": {"url": VIDEO_URL},
    }


def test_generate_stream_text_sends_video_part():
    provider, client = _make_provider()

    chunks = list(
        provider.generate_stream_text(
            prompt="describe this", llm_model="test-model", video_url=VIDEO_URL
        )
    )

    assert chunks == ["ok"]
    assert _part_types(client) == ["text", "video_url"]
    assert client.captured["stream"] is True


def test_structured_response_sends_video_part():
    provider, client = _make_provider(structured=True)

    response = provider.structured_response(
        prompt="describe this",
        response_model=_Answer,
        llm_model="test-model",
        video_url=VIDEO_URL,
    )

    assert response.answer == "ok"
    assert _part_types(client) == ["text", "video_url"]
