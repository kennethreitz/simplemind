import pytest

from simplemind.providers import Anthropic, Gemini, OpenAI, Groq, Ollama, Amazon

import simplemind as sm


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
