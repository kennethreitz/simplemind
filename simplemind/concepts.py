from pydantic import BaseModel
from typing import Dict, Any
from simplemind.plugins.base import BasePlugin


class Context(BaseModel):
    plugins: Dict[str, BasePlugin] = {}

    def add_plugin(self, name: str, plugin: BasePlugin):
        self.plugins[name] = plugin

    def execute_plugin(self, name: str, *args, **kwargs):
        if name in self.plugins:
            return self.plugins[name].execute(self, *args, **kwargs)
        else:
            raise ValueError(f"Plugin '{name}' not found in context.")
