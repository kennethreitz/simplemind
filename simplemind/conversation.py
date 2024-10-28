class Conversation:
    """A class to manage conversation state with an AI model."""

    def __init__(self, client):
        self.client = client
        self.messages = []

    def add_message(self, message, role="user"):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": message})
        return self

    def send(self, message=None, **kwargs):
        """Send the conversation history (and optionally a new message) to the AI."""
        if message:
            self.add_message(message)

        response = self.client.message(message_history=self.messages, **kwargs)

        # Add the AI's response to the conversation history
        if isinstance(response.text, str):
            self.add_message(response.text, role="assistant")

        return response
