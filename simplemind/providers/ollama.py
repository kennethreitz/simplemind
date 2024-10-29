import ollama as ol
import instructor
from openai import OpenAI

from ._base import BaseProvider
from ..settings import settings

PROVIDER_NAME = "ollama"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 60


class Ollama(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL
    TIMEOUT = DEFAULT_TIMEOUT

    def __init__(self, host_url: str = None):
        self.host_url = host_url or settings.OLLAMA_HOST_URL

    @property
    def client(self):
        """The raw Ollama client."""
        if not self.host_url:
            raise ValueError("No ollama host url provided")
        return ol.Client(timeout=self.TIMEOUT, host=self.host_url)

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        return instructor.from_openai(
            OpenAI(
                base_url=f"{self.host_url}/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )

    def send_conversation(self, conversation: "Conversation"):
        """Send a conversation to the Ollama API."""
        from ..models import Message

        messages = [
            {"role": msg.role, "content": msg.text} for msg in conversation.messages
        ]
        response = self.client.chat(
            model=conversation.llm_model or DEFAULT_MODEL, messages=messages
        )
        assistant_message = response.get("message")

        # Create and return a properly formatted Message instance
        return Message(
            role="assistant",
            text=assistant_message.get("content"),
            raw=response,
            llm_model=conversation.llm_model or DEFAULT_MODEL,
            llm_provider=PROVIDER_NAME,
        )

    def structured_response(self, prompt, response_model, *, llm_model: str, **kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.structured_client.chat.completions.create(
            messages=messages, model=llm_model, response_model=response_model, **kwargs
        )
        return response

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat(messages=messages, model=llm_model)

        return response.get("message").get("content")
