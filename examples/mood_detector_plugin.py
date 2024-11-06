import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from rich.console import Console
from _context import simplemind as sm

nltk.download("vader_lexicon")

console = Console()


class MoodDetectorPlugin(sm.BasePlugin):
    model_config = {"arbitrary_types_allowed": True}
    analyzer: SentimentIntensityAnalyzer = None

    def __init__(self):
        super().__init__()
        # Initialize sentiment analyzer from nltk
        self.analyzer = SentimentIntensityAnalyzer()

    def detect_mood(self, text):
        # Analyze the sentiment of the given text
        scores = self.analyzer.polarity_scores(text)

        # Print sentiment analysis details with colors
        console.print("\n[bold]Sentiment Analysis:[/bold]")
        console.print(f"Text: [italic]{text}[/italic]")
        console.print("\nScores:")
        console.print(f"🟢 Positive: [green]{scores['pos']:.3f}[/green]")
        console.print(f"🔴 Negative: [red]{scores['neg']:.3f}[/red]")
        console.print(f"⚪ Neutral: [blue]{scores['neu']:.3f}[/blue]")
        console.print(f"📊 Compound: [yellow]{scores['compound']:.3f}[/yellow]\n")

        if scores["compound"] >= 0.5:
            console.print("Overall Mood: [green]positive[/green] 😊")
            return "positive"
        elif scores["compound"] <= -0.5:
            console.print("Overall Mood: [red]negative[/red] 😢")
            return "negative"
        else:
            console.print("Overall Mood: [blue]neutral[/blue] 😐")
            return "neutral"

    def pre_send_hook(self, conversation: sm.Conversation):
        # Get the last user message to analyze its mood
        last_message = conversation.get_last_message(role="user")
        if last_message:
            mood = self.detect_mood(last_message.text)
            # Adjust AI response style based on the detected mood
            if mood == "positive":
                tone_message = (
                    "The user seems cheerful. Respond with enthusiasm and positivity."
                )
            elif mood == "negative":
                tone_message = "The user seems to be in a low mood. Respond with empathy and warmth."
            else:
                tone_message = "The user seems neutral. Respond with a balanced tone."

            # Inject the tone adjustment message as a system prompt
            conversation.add_message(role="system", text=tone_message)


# Create a conversation and add the plugin
conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")
conversation.add_plugin(MoodDetectorPlugin())

# Add a user message and send the conversation
conversation.add_message(role="user", text="I'm having a really rough day.")
response = conversation.send()

console.print(f"*{ response.text }*")
