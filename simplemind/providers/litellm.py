from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Iterator, Type, TypeVar

import instructor
from pydantic import BaseModel

from ..logging import logger
from ..settings import settings
from ._base import BaseProvider
from ._base_tools import BaseTool
from .openai import OpenAITool

if TYPE_CHECKING:
    from ..models import Conversation, Message

T = TypeVar("T", bound=BaseModel)


class LiteLLMTool(OpenAITool):
    """LiteLLM returns OpenAI-shaped tool calls for every upstream provider."""


class LiteLLM(BaseProvider):
    """Any provider LiteLLM supports, addressed as `<provider>/<model>`.

    `llm_model` picks the upstream (`anthropic/claude-sonnet-4-5`,
    `bedrock/...`, `vertex_ai/gemini-2.5-flash`, `azure/<deployment>`,
    `openrouter/...`, or `litellm_proxy/<alias>` for a LiteLLM proxy), and
    LiteLLM reads that provider's credentials from its usual environment
    variables. `LITELLM_API_KEY` and `LITELLM_API_BASE` are optional and only
    sent when set, e.g. to point at a LiteLLM proxy.
    """

    NAME = "litellm"
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_MAX_TOKENS = None
    # Drop parameters the selected upstream model does not accept instead of
    # failing the request (e.g. `seed` on Anthropic).
    DEFAULT_KWARGS = {"drop_params": True}
    supports_streaming = True

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key or settings.get_api_key(self.NAME)
        self.api_base = api_base or settings.LITELLM_API_BASE

    @cached_property
    def client(self):
        """The `litellm` module, whose `completion` mirrors the OpenAI API."""
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "Please install the `litellm` package: `pip install 'simplemind[litellm]'`"
            ) from exc
        litellm.suppress_debug_info = True
        return litellm

    @cached_property
    def structured_client(self):
        """LiteLLM's completion patched with Instructor."""
        return instructor.from_litellm(self.client.completion)

    def _request_kwargs(self, **kwargs) -> dict[str, Any]:
        """Merge defaults, credentials (only when set) and call kwargs."""
        credentials = {}
        if self.api_key:
            credentials["api_key"] = self.api_key
        if self.api_base:
            credentials["api_base"] = self.api_base.rstrip("/")
        return {**self.DEFAULT_KWARGS, **credentials, **kwargs}

    @staticmethod
    def _user_message(
        prompt: str, image_url: str | None = None, video_url: str | None = None
    ) -> list[dict]:
        """Build a user message, adding image / video parts (url or base64) if provided."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        if video_url:
            content.append({"type": "video_url", "video_url": {"url": video_url}})
        return [{"role": "user", "content": content}]

    @logger
    def send_conversation(
        self,
        conversation: "Conversation",
        tools: list[Callable | BaseTool] | None = None,
        **kwargs,
    ) -> "Message":
        """Send a conversation through LiteLLM."""
        from ..models import Message

        formatted_messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]

        converted_tools = self.make_tools(tools)
        tools_config = (
            [t.get_input_schema() for t in converted_tools] if tools else None
        )

        request_kwargs = self._request_kwargs(
            **kwargs,
            model=conversation.llm_model or self.DEFAULT_MODEL,
            messages=formatted_messages,
        )

        if tools_config:
            request_kwargs["tools"] = tools_config

        response = self.client.completion(**request_kwargs)

        # Run requested tools and send their results back until the model answers.
        while response.choices[0].message.tool_calls:
            for tool in converted_tools:
                tool.handle(response, formatted_messages)
                if tool.is_executed():
                    response = self.client.completion(**request_kwargs)
                    tool.reset_result()

        final_message = response.choices[0].message.content

        return Message(
            role="assistant",
            text=final_message or "",
            raw=response,
            llm_model=conversation.llm_model or self.DEFAULT_MODEL,
            llm_provider=self.NAME,
        )

    @logger
    def structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        llm_model: str | None = None,
        image_url: str | None = None,
        video_url: str | None = None,
        **kwargs,
    ) -> T:
        """Get a structured response through LiteLLM."""
        response = self.structured_client.chat.completions.create(
            **self._request_kwargs(
                **kwargs,
                messages=self._user_message(prompt, image_url, video_url),
                model=llm_model or self.DEFAULT_MODEL,
                response_model=response_model,
            )
        )
        return response_model.model_validate(response)

    @logger
    def generate_text(
        self,
        prompt: str,
        *,
        llm_model: str | None = None,
        image_url: str | None = None,
        video_url: str | None = None,
        **kwargs,
    ) -> str:
        """Generate text through LiteLLM."""
        response = self.client.completion(
            **self._request_kwargs(
                **kwargs,
                messages=self._user_message(prompt, image_url, video_url),
                model=llm_model or self.DEFAULT_MODEL,
            )
        )
        return response.choices[0].message.content

    @logger
    def generate_stream_text(
        self,
        prompt: str,
        *,
        llm_model: str | None = None,
        image_url: str | None = None,
        video_url: str | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate streaming text through LiteLLM.

        Yields chunks of text as they are generated by the model.
        """
        response = self.client.completion(
            **self._request_kwargs(
                **kwargs,
                messages=self._user_message(prompt, image_url, video_url),
                model=llm_model or self.DEFAULT_MODEL,
                stream=True,
            )
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @cached_property
    def tool(self) -> Type[BaseTool]:
        """The tool implementation for LiteLLM."""
        return LiteLLMTool
