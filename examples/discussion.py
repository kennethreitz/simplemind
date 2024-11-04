import time
from typing import List, Tuple

from rich.console import Console
from rich.markdown import Markdown

from _context import sm


class MultiAIConversation:
    """Orchestrates conversations between multiple AI models."""

    MODEL_SESSIONS = {
        "GPT-4o": sm.Session(
            llm_provider="openai",
            llm_model="gpt-4o",
        ),
        "Grok-Beta": sm.Session(
            llm_provider="xai",
            llm_model="grok-beta",
        ),
        "Claude-3.5-Sonnet": sm.Session(
            llm_provider="anthropic",
            llm_model="claude-3-5-sonnet-20241022",
        ),
    }

    def __init__(self, topic: str, turns_per_model: int = 1, max_rounds: int = 5):

        self.topic = topic
        self.turns_per_model = turns_per_model
        self.max_rounds = max_rounds
        self.conversation_history: List[Tuple[str, str]] = []
        self.console = Console()
        self.user_name = "Kenneth Reitz"

    def _format_system_prompt(self, ai_name: str) -> str:
        """Creates a system prompt for each AI model."""
        return f"""You are {ai_name}. You are participating in a thoughtful discussion with other AI models about {self.topic}.

Rules:
1. Be concise but insightful (keep responses under 140 words)
2. Build upon previous points made in the conversation
3. Ask questions to deepen the discussion when appropriate
4. Stay on topic while maintaining your unique perspective
5. Be respectful of other viewpoints while maintaining your distinct voice

Current discussion topic: {self.topic}"""

    def _create_conversation(
        self, session: sm.Session, ai_name: str
    ) -> sm.Conversation:
        """Creates a new conversation with appropriate context for an AI model."""
        conv = session.create_conversation()

        # Add system prompt
        conv.add_message(role="user", text=self._format_system_prompt(ai_name))

        # Add conversation history
        for speaker, message in self.conversation_history[-3:]:  # Last 3 messages
            conv.add_message(role="user", text=f"{speaker} said: {message}")

        return conv

    def _print_response(self, ai_name: str, response: str):
        """Pretty prints an AI response using Rich."""
        self.console.print(f"\n[bold blue]{ai_name}[/bold blue]:")
        self.console.print(Markdown(response))
        # Store in history
        self.conversation_history.append((ai_name, response))

    def _get_user_input(self) -> str:
        """Gets input from the user for the discussion."""
        self.console.print("\n[bold green]Your turn! Share your thoughts:[/bold green]")
        user_response = input("> ")
        self._print_response(self.user_name, user_response)
        return user_response

    def run_conversation(self):
        """Runs the multi-AI conversation."""
        # Get initial thoughts from the human
        self.console.print(
            f"\n[bold green]Start the discussion about {self.topic}:[/bold green]"
        )
        self._get_user_input()

        for round_num in range(self.max_rounds):
            self.console.print(f"\n[bold green]Round {round_num + 1}[/bold green]")

            # Let all AI models respond
            for model_name, session in self.MODEL_SESSIONS.items():
                for turn in range(self.turns_per_model):
                    conversation = self._create_conversation(session, model_name)

                    # Add the prompt (simplified since human always starts)
                    prompt = f"Continue the discussion about {self.topic}, responding to the previous points made."
                    conversation.add_message(role="user", text=prompt)

                    # Get and print response
                    response = conversation.send()
                    self._print_response(model_name, response.text)

                    # Small delay to prevent rate limiting
                    time.sleep(1)

            # Then get user input at the end of the round
            self._get_user_input()

            # Optional: Add a separator between rounds
            self.console.print("\n" + "-" * 50)


def have_ai_discussion(turns_per_model: int = 1, max_rounds: int = 3):
    """Convenience function to start an AI discussion."""
    # Get topic from user
    print("\nWhat topic would you like to discuss?")
    topic = input("> ")

    debate = MultiAIConversation(
        topic=topic, turns_per_model=turns_per_model, max_rounds=max_rounds
    )

    print(f"\nStarting AI discussion about: {topic}")
    print("=" * 50)

    debate.run_conversation()


# Example usage
if __name__ == "__main__":
    have_ai_discussion(turns_per_model=1, max_rounds=5)
