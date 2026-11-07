import logging
import os
import subprocess
import sys
from unittest import mock

import pytest

import simplemind as sm
from simplemind.logging import log_verbose


@pytest.fixture
def provider():
    """A fake provider, so no API key or network is needed."""
    fake = mock.MagicMock()
    fake.NAME = "fake"
    with mock.patch("simplemind.find_provider", return_value=fake):
        yield fake


@pytest.fixture
def verbose_on(monkeypatch):
    monkeypatch.setattr(sm.settings, "VERBOSE", True)


def messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_verbose_is_off_by_default():
    assert sm.settings.VERBOSE is False


def test_verbose_off_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(sm.settings, "VERBOSE", False)
    with caplog.at_level(logging.DEBUG, logger="simplemind"):
        log_verbose("should not appear")
    assert caplog.records == []


def test_verbose_on_logs_at_info(verbose_on, caplog):
    with caplog.at_level(logging.INFO, logger="simplemind"):
        log_verbose("hello")
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("simplemind", logging.INFO, "hello")
    ]


def test_per_call_flag_overrides_settings(monkeypatch, caplog):
    monkeypatch.setattr(sm.settings, "VERBOSE", False)
    with caplog.at_level(logging.INFO, logger="simplemind"):
        log_verbose("on for this call", enabled=True)
    assert messages(caplog) == ["on for this call"]


def test_per_call_false_silences_global_on(verbose_on, caplog):
    with caplog.at_level(logging.INFO, logger="simplemind"):
        log_verbose("silenced", enabled=False)
    assert caplog.records == []


def test_generate_text_logs_provider_model_and_response(verbose_on, provider, caplog):
    provider.generate_text.return_value = "the answer"
    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_text("secret prompt", llm_model="m1")
    assert "generate_text provider=fake model=m1 stream=False" in messages(caplog)
    assert "generate_text response: the answer" in messages(caplog)


def test_generate_text_does_not_log_the_prompt(verbose_on, provider, caplog):
    provider.generate_text.return_value = "ok"
    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_text("secret prompt")
    assert all("secret prompt" not in m for m in messages(caplog))


def test_generate_text_logs_default_model_when_none(verbose_on, provider, caplog):
    provider.generate_text.return_value = "ok"
    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_text("hi")
    assert "generate_text provider=fake model=default stream=False" in messages(caplog)


def test_generate_text_per_call_verbose_is_not_sent_to_provider(provider):
    provider.generate_text.return_value = "ok"
    sm.generate_text("hi", verbose=True)
    assert "verbose" not in provider.generate_text.call_args.kwargs


def test_generate_text_per_call_verbose_logs_without_global(
    monkeypatch, provider, caplog
):
    monkeypatch.setattr(sm.settings, "VERBOSE", False)
    provider.generate_text.return_value = "ok"
    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_text("hi", verbose=True)
    assert "generate_text response: ok" in messages(caplog)


def test_generate_text_stream_yields_chunks_unchanged(provider):
    provider.supports_streaming = True
    provider.generate_stream_text.return_value = iter(["Hel", "lo"])
    assert list(sm.generate_text("hi", stream=True, verbose=False)) == ["Hel", "lo"]


def test_generate_text_stream_logs_full_text_after_consumption(
    verbose_on, provider, caplog
):
    provider.supports_streaming = True
    provider.generate_stream_text.return_value = iter(["Hel", "lo"])
    with caplog.at_level(logging.INFO, logger="simplemind"):
        stream = sm.generate_text("hi", stream=True)
        assert "generate_text stream response" not in " ".join(messages(caplog))
        assert "".join(stream) == "Hello"
    assert "generate_text stream response: Hello" in messages(caplog)


def test_generate_text_stream_logs_request_before_consumption(
    verbose_on, provider, caplog
):
    provider.supports_streaming = True
    provider.generate_stream_text.return_value = iter(["x"])
    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_text("hi", stream=True)
    assert "generate_text provider=fake model=default stream=True" in messages(caplog)


def test_generate_data_logs_response_model(verbose_on, provider, caplog):
    class Answer:
        pass

    with caplog.at_level(logging.INFO, logger="simplemind"):
        sm.generate_data("hi", response_model=Answer)
    msgs = messages(caplog)
    assert "generate_data provider=fake model=default response_model=Answer" in msgs
    assert any(m.startswith("generate_data response: ") for m in msgs)


def test_conversation_send_logs_and_accepts_per_call_flag(provider, caplog):
    conv = sm.create_conversation(llm_model="m2", llm_provider="fake")
    conv.add_message("user", "hi")
    response = mock.MagicMock(text="reply", meta={})
    provider.send_conversation.return_value = response
    with mock.patch("simplemind.models.find_provider", return_value=provider):
        with caplog.at_level(logging.INFO, logger="simplemind"):
            conv.send(verbose=True)
    msgs = messages(caplog)
    assert "send provider=fake model=m2" in msgs
    assert "send response: reply" in msgs


def test_verbose_does_not_touch_root_logger_handlers(verbose_on):
    root = logging.getLogger()
    before = list(root.handlers)
    log_verbose("hi")
    assert root.handlers == before


def test_verbose_adds_stderr_handler_when_nothing_configured(tmp_path):
    """In a fresh process with no logging config, VERBOSE output still shows."""
    env = {**os.environ, "VERBOSE": "1"}
    code = (
        "import logging\n"
        "from simplemind.logging import log_verbose\n"
        "log_verbose('visible')\n"
        "print('root_handlers', len(logging.getLogger().handlers))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=True,
    )
    assert "visible" in result.stderr
    assert "root_handlers 0" in result.stdout
