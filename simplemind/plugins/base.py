from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlugin(ABC):
    """Base class for all SimpleMind plugins."""

    def __init__(self):
        self.is_enabled = True

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context and return modified context.

        Args:
            context: The current conversation context

        Returns:
            Modified context dictionary
        """
        pass

    def enable(self):
        """Enable the plugin."""
        self.is_enabled = True

    def disable(self):
        """Disable the plugin."""
        self.is_enabled = False
