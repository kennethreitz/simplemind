from .base import BaseChain


class ReverseTextChain(BaseChain):
    def run(self, input_data):
        return input_data[::-1]
