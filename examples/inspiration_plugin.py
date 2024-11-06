import random
from _context import simplemind as sm


class InspirationPlugin(sm.BasePlugin):
    # Define inspirations as a class variable
    inspirations: list[str] = [
        "The only limit to our realization of tomorrow is our doubts of today.",
        "Imagine beyond the edges of what you know.",
        "What if the stars could speak? What stories would they tell?",
        "Creativity is intelligence having fun.",
        "Think not only with your mind but with your heart.",
        "Let every answer be a doorway to another question.",
        "The universe is in constant dialogue with those who listen.",
    ]

    def get_inspiration(self):
        # Randomly select an inspirational quote or prompt
        return random.choice(self.inspirations)

    def pre_send_hook(self, conversation: sm.Conversation):
        # Inject an inspirational message as a system prompt
        inspiration = self.get_inspiration()
        conversation.add_message(role="system", text=inspiration)


# Create a conversation and add the plugin
conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")
conversation.add_plugin(InspirationPlugin())

# Add a user message and send the conversation
conversation.add_message(role="user", text="Tell me something inspiring.")
response = conversation.send()
print(response.text)
