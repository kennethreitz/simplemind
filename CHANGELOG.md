Release History
===============

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
