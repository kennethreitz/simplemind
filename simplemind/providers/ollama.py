import os

from ollama import Client as BaseOllama

from .base import BaseClientProvider
from ..core.models import AIResponse
from ..conversation import Conversation

TIMEOUT = 60

class Ollama(BaseClientProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.login()
        self.conversation = []

    def login(self):
        """Initialize Ollama client, with Instructor enabled."""
        if not os.environ.get('OLLAMA_HOST_URL'):
           raise ValueError("Please set the OLLAMA_HOST_URL environment variable")

        if not os.environ.get('OLLAMA_MODEL'):
            raise ValueError("Please set the OLLAMA_MODEL environment variable")
        else:
            self.model = os.environ.get('OLLAMA_MODEL')

        self.client = BaseOllama(
            timeout=TIMEOUT,
            host=os.environ.get('OLLAMA_HOST_URL'))
        assert self.test_connection()

    @property
    def available_models(self):
        """Returns the available models from the Ollama client."""

        def gen():
            for model in self.client.list().get('models'):
                yield model

        return [g for g in gen()]

    def test_connection(self):
        """Test the connection to Ollama. Returns True if successful."""

        return bool(len(self.available_models))

    def generate_text(self, prompt, *, response_model=False, **kwargs):
        use_instructor = bool(response_model)

        client = self.instructor_client if use_instructor else self.client

        # Parameters for the Ollama client.
        params = {
            "prompt": prompt,
            "model": self.model,
        }
        params.update(kwargs)

        if use_instructor:
            params["response_model"] = response_model

        # Make the request to Ollama.
        completion = client.generate(**params)
        if use_instructor:
            return completion.model_dump()

        else:
            return AIResponse(
                response=completion,
                text=completion.get('response'),
            )

    def message(self, message=None, message_history=None, response_model=False, **kwargs):
        """Generates a response from the Ollama client."""
        use_instructor = bool(response_model)

        client = self.instructor_client if use_instructor else self.client

        # Parameters for the Ollama client.
        all_messages = []
        if message_history:
            all_messages.extend(message_history)
        if message:
            all_messages.append({'role': 'user', 'content': message})
        params = {
            "messages": all_messages,
            "model": self.model,
        }
        params.update(kwargs)

        if use_instructor:
            params["response_model"] = response_model

        # Make the request to Ollama.
        completion = client.chat(**params)
        if use_instructor:
            return completion.model_dump()

        else:
            return AIResponse(
                response=completion,
                text=completion.get('message').get('content'),
            )

    def start_conversation(self):
        return Conversation(self)
