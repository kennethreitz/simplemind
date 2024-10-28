import os
import logging

class BaseClientProvider:

    def __init__(self, *, api_key=None, api_key_environ_name=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self._api_key_environ_name = api_key_environ_name

        # Load API key from environment if not provided
        self._api_key = api_key or self._load_from_environ()

    def _load_from_environ(self):
        if self._api_key_environ_name:
            return os.environ.get(self._api_key_environ_name)
        return None

    def test_connection(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def generate_response(self, request):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def health_check(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def available_models(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def features(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")
