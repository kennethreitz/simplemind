from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePlugin(ABC):
    """Base class for all SimpleMind plugins."""

    def __init__(self):
        self.enabled: bool = True
        self.name: str = self.__class__.__name__

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context and return modified context.

        Args:
            context: The current conversation context

        Returns:
            Modified context dictionary
        """
        pass

    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False

    @property
    def is_enabled(self) -> bool:
        return self.enabled
