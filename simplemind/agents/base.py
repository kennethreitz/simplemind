from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def decide(self, context, *args, **kwargs):
        pass
