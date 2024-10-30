import simplemind as sm


class LoggingPlugin(sm.BasePlugin):
    def pre_send_hook(self, conversation):
        print(f"Sending conversation with {len(conversation.messages)} messages")

    def add_message_hook(self, conversation, message):
        print(f"Adding message to conversation: {message.text}")

    def cleanup_hook(self, conversation):
        print(f"Cleaning up conversation with {len(conversation.messages)} messages")

    def initialize_hook(self, conversation):
        print("Initializing conversation")

    def post_send_hook(self, conversation, response):
        print(f"Received response: {response.text}")


with sm.create_conversation() as conversation:
    # Add the logging plugin.
    conversation.add_plugin(LoggingPlugin())

    # Add a message to the conversation.
    conversation.add_message("user", "Hello!", meta={})

    # Send the conversation.
    response = conversation.send()

print(f"Response: {response.text}")
