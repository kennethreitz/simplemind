import os

class BaseClientProvider:

    def __init__(self, *, api_key=None, environ_name=None):
        self._environ_name = environ_name
        # TODO: reverse order, potentially?
        self._api_key = api_key or self._load_from_environ()

    def _load_from_environ(self):
        if self._environ_name:
            self.api_key = os.environ.get(self._environ_name)
        else:
            self.api_key = None

    def test_connection(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def generate_response(self, request):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def health_check(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    def available_models(self):
        raise NotImplementedError("This method must be implemented by the AI provider client.")

    # TODO: logging provider.
    #
