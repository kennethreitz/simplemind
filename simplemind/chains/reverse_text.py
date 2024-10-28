from .base import BaseChain


class ReverseTextChain(BaseChain):
    """Chain that reverses input text.

    This chain takes a text input and returns it reversed. For example,
    "hello" becomes "olleh".
    """

    def run(self, input_data: str) -> str:
        """Reverse the input text.

        Args:
            input_data: The text to reverse.

        Returns:
            The reversed text.

        Raises:
            TypeError: If input_data is not a string.
        """
        if not isinstance(input_data, str):
            raise TypeError("Input must be a string")
        return input_data[::-1]
