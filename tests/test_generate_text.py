import pytest

from simplemind.providers import Anthropic, Gemini, OpenAI, Groq, Ollama, Amazon


@pytest.mark.parametrize(
    "provider_cls",
    [
        Anthropic,
        Gemini,
        OpenAI,
        Groq,
        Ollama,
        # Amazon,
    ],
)
def test_generate_text(provider_cls):
    provider = provider_cls()
    prompt = "What is 2+2?"

    response = provider.generate_text(prompt=prompt, llm_model=provider.DEFAULT_MODEL)

    assert isinstance(response, str)
    assert len(response) > 0
