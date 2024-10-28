from .base import BasePlugin


class BasicMemoryPlugin(BasePlugin):
    def __init__(self):
        self.memory = []

    def execute(self, context, message):
        self.memory.append(message)
        return self.memory
