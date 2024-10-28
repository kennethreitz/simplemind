from .base import BasePlugin


class KVPlugin(BasePlugin):
    def __init__(self):
        self.store = {}

    def execute(self, context, key, value):
        self.store[key] = value
        return self.store
