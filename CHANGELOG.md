Release History
===============

## 0.2.3 (2024-11-04)

- Remove default max-tokens for OpenAI provider.

## 0.2.3 (2024-11-03)

- Update default model for Amazon provider.
- Improved logging to handle streaming functions.

## 0.2.2 (2024-11-02)

- Add streaming support (set `stream=True` to `generate_text`).
- `conv.prepend_system_message` now uses system role by default.
- Add `provider.supports_streaming` property.
- Add `provider.supports_structured_response` property.
- General improvements.

## 0.2.1 (2024-11-01)

- Add `cached_property` to Amazon provider.

## 0.2.0 (2024-11-01)

- Add Amazon Bedrock provider.
- Make all provider optional dependencies. Use `$ pip install 'simplemind[full]'` to install all providers.
- General improvements.

## 0.1.7 (2024-11-01)

- Add `logger` decorator.
- Add `sm.enable_logfire()` function.
- General improvements.

## 0.1.6 (2024-10-31)

- Add `sm.Plugin` syntax sugar.
- Improvements to Anthropic provider, related to max tokens.
- General improvements.
- Add tests for structured response.
- Add `llm_model` to `structured_response`.

## 0.1.5 (2024-10-31)

- Add Gemini provider.
- Add structured response to Gemini provider.
- Support for Python 3.10.

## 0.1.4 (2024-10-30)

- Introduce `Session` class to manage repeatability.
- General improvements.

## 0.1.3 (2024-10-30)

- Make Conversation a context manager.
- Add more robust conversation plugin hooks — replace `send_hook` with `pre_send_hook` and `post_send_hook`.
- Change plugin hooks to try/except NotImplementedError.
- Implement 'did you mean' with provider names. Can do this eventually with model names, as well.

## 0.1.2 (2024-10-29)

- Add ollama provider.

## 0.1.1 (2024-10-29)

- Fix Groq provider.

## 0.1.0 (2024-10-29)

- Initial release.
