import time
from _context import simplemind as sm


class ConversationDisplay(sm.BasePlugin):
    def post_send_hook(self, conversation, response):
        # Simple print output instead of Rich formatting
        print(f"\n{conversation.llm_provider}:")
        print(f"{response.text.strip()}\n")


def four_way_conversation(topic: str, rounds: int = 3):
    # Create conversations for four different AIs
    with (
        sm.create_conversation(llm_provider="anthropic") as claude_conv,
        sm.create_conversation(llm_model="gpt-4", llm_provider="openai") as gpt4_conv,
        sm.create_conversation(
            llm_model="llama3.2", llm_provider="ollama"
        ) as llama_conv,
        sm.create_conversation(llm_provider="groq") as groq_conv,
    ):
        # Add display plugin to each conversation
        display = ConversationDisplay()
        for conv in [claude_conv, gpt4_conv, llama_conv, groq_conv]:
            conv.add_plugin(display)

        # Initial prompt
        print(f"\nTopic: {topic}\n")

        # Start with Claude
        claude_conv.add_message(
            "user",
            f"Share your thoughts on this topic: {topic}. Keep your response concise.",
            meta={},
        )
        last_response = claude_conv.send()

        # Continue the conversation
        for _ in range(rounds):
            for conv in [llama_conv, gpt4_conv, groq_conv, claude_conv]:
                # Add a small delay between responses
                time.sleep(1)

                # Each AI responds to the previous statement
                conv.add_message(
                    "user",
                    f"Respond to this perspective from another AI about {topic}: "
                    f"{last_response.text}\nKeep your response concise and add your own insights.",
                    meta={},
                )
                last_response = conv.send()


if __name__ == "__main__":
    topic = "A new platform for AI and humans to co-create together. What would it look like? Discuss."
    print("\nStarting a four-way AI conversation...\n")
    four_way_conversation(topic)
    print("\nConversation ended.\n")
