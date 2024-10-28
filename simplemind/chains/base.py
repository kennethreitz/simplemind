from abc import ABC, abstractmethod


class BaseChain(ABC):
    @abstractmethod
    def run(self, input_data):
        pass
