import time

import simplemind as sm


class ConversationPlugin(sm.BasePlugin):
    def post_send_hook(self, conversation, response):
        # Print the LLM model and the response text.
        print(f"{conversation.llm_model}:\n{response.text.strip()}\n\n------------\n")


def have_conversation(rounds: int = 3):
    # Create two conversations - one for each AI
    with (
        sm.create_conversation(
            llm_model="claude-3-5-sonnet-20241022", llm_provider="anthropic"
        ) as claude_conv,
        sm.create_conversation(
            llm_model="llama3.2", llm_provider="ollama"
        ) as llama_conv,
    ):

        # Add our plugin to both
        plugin = ConversationPlugin()
        claude_conv.add_plugin(plugin)
        llama_conv.add_plugin(plugin)

        # Start the conversation
        prompt = "What do you think about the future of artificial intelligence? Please keep your response brief."
        claude_conv.add_message("user", prompt, meta={})
        claude_response = claude_conv.send()

        # Have them discuss back and forth
        for _ in range(rounds):
            # Llama responds to Claude
            llama_conv.add_message(
                "user",
                f"Respond to this statement from another AI: {claude_response.text}",
                meta={},
            )
            llama_response = llama_conv.send()

            time.sleep(1)  # Add a small delay between responses

            # Claude responds to Llama
            claude_conv.add_message(
                "user",
                f"Respond to this statement from another AI: {llama_response.text}",
                meta={},
            )
            claude_response = claude_conv.send()

            time.sleep(1)


if __name__ == "__main__":
    print("Starting AI conversation...\n")
    have_conversation()
    print("\nConversation ended.")
