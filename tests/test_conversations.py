import pytest

import simplemind as sm
from simplemind.providers import Amazon, Anthropic, Gemini, Groq, Ollama, OpenAI


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
