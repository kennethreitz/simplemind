from .base import BasePlugin


class KVPlugin(BasePlugin):
    def __init__(self):
        self.store = {}

    def process(self, key: str, value=None):
        """
        Get or set a value in the key-value store.
        If value is None, returns the value for the key.
        If value is provided, sets the value for the key and returns it.
        """
        if value is None:
            return self.store.get(key)

        self.store[key] = value
        return value
