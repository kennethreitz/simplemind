import ollama as ol

from ._base import BaseProvider
from ..settings import settings

PROVIDER_NAME = "ollama"
DEFAULT_MODEL = "llama3.2"
TIMEOUT = 60
NOT_IMPLEMENTED_REASON = """
# TODO: instructor does not natively support ollama.
# Alternate python dependency may be required
"""

class Ollama(BaseProvider):
    NAME = PROVIDER_NAME
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, host_url: str = None):
        self.host_url = host_url or settings.OLLAMA_HOST_URL

    @property
    def client(self):
        """The raw Ollama client."""
        if not self.host_url:
            raise ValueError("No ollama host url provided")
        return ol.Client(
            timeout=TIMEOUT,
            host=self.host_url)

    @property
    def structured_client(self):
        """A client patched with Instructor."""
        raise NotImplementedError(NOT_IMPLEMENTED_REASON)

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

    def structured_response(self, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_REASON)

    def generate_text(self, prompt, *, llm_model):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat(
            messages=messages, model=llm_model
        )

        return response.get("message").get("content")
