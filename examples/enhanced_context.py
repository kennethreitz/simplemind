from datetime import datetime, timedelta
import logging
import sqlite3
from typing import List
import re

import spacy
from contextlib import contextmanager

from _context import simplemind as sm

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

DB_PATH = "enhanced_context.db"


class EnhancedContextPlugin(sm.BasePlugin):
    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__()
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error(
                "Failed to load spaCy model. Please install it using: python -m spacy download en_core_web_sm"
            )
            raise

        # Initialize database
        self.init_db()
        self.logger.info(f"EnhancedContextPlugin initialized with database: {DB_PATH}")

        # Load identity from database
        self.personal_identity = None
        self.load_identity()

        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("averaged_perceptron_tagger")
        except LookupError:
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(DB_PATH)
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initialize the database with proper schema"""
        with self.get_connection() as conn:
            # Create memory table for entities
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    entity TEXT PRIMARY KEY,
                    last_mentioned TIMESTAMP,
                    mention_count INTEGER DEFAULT 1
                )
            """
            )

            # Create identity table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identity (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    last_updated TIMESTAMP
                )
            """
            )

    def store_entity(self, entity: str) -> None:
        """Store or update entity mention with error handling"""
        try:
            with self.get_connection() as conn:
                # Modified to store datetime in SQLite format
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT INTO memory (entity, last_mentioned, mention_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(entity) DO UPDATE SET
                        last_mentioned = ?,
                        mention_count = mention_count + 1
                    """,
                    (entity, now, now),
                )
                conn.commit()  # Added explicit commit
                self.logger.info(f"Stored entity: {entity}")
        except sqlite3.Error as e:
            self.logger.error(f"Database error while storing entity {entity}: {e}")

    def retrieve_recent_entities(self, days: int = 7) -> List[str]:
        """Retrieve recently mentioned entities with frequency"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                # Modified query to handle datetime strings properly
                cur.execute(
                    """
                    SELECT entity, mention_count
                    FROM memory
                    WHERE last_mentioned >= datetime('now', ?, 'localtime')
                    ORDER BY mention_count DESC, last_mentioned DESC
                    LIMIT 5
                    """,
                    (f"-{days} days",),
                )

                entities = [(row[0], row[1]) for row in cur.fetchall()]
                self.logger.info(f"Retrieved recent entities: {entities}")
                return entities
        except sqlite3.Error as e:
            self.logger.error(f"Database error while retrieving entities: {e}")
            return []

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities with improved filtering"""
        doc = self.nlp(text)
        entities = []

        # Define important entity types
        important_types = {
            "PERSON",
            "ORG",
            "GPE",
            "NORP",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
        }

        for ent in doc.ents:
            if (
                ent.label_ in important_types
                and len(ent.text.strip()) > 1  # Avoid single characters
                and not ent.text.isnumeric()
            ):  # Avoid pure numbers
                entities.append(ent.text.strip())

        return list(set(entities))  # Remove duplicates

    def format_context_message(
        self, entities: List[tuple], include_identity: bool = True
    ) -> str:
        """Format context message more naturally"""
        context_parts = []

        # Add identity context if available and requested
        if include_identity and self.personal_identity:
            context_parts.append(f"You are speaking with {self.personal_identity}")

        # Add entity context if available
        if entities:
            # Format entities with their mention counts
            entity_strings = [
                f"{entity} ({'mentioned multiple times' if count > 1 else 'mentioned recently'})"
                for entity, count in entities
            ]

            context_parts.append(
                f"Previously discussed topic{'s' if len(entity_strings) > 1 else ''}: "
                + (
                    ", ".join(entity_strings[:-1]) + f" and {entity_strings[-1]}"
                    if len(entity_strings) > 1
                    else entity_strings[0]
                )
            )

        return ". ".join(context_parts) + ("." if context_parts else "")

    def extract_identity(self, text: str) -> str | None:
        """Extract identity statements like 'I am X'"""
        text = text.lower().strip()
        if text.startswith("i am ") or text.startswith("my name is "):
            identity = text.replace("i am ", "").replace("my name is ", "").strip()
            return identity if identity else None
        return None

    def is_identity_question(self, text: str) -> bool:
        """Use NLTK to detect identity questions"""
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)

        # Extract key words and patterns
        words = set(tokens)
        has_question_word = any(word in ["who", "what"] for word in words)
        has_identity_term = any(word in ["i", "me", "my", "name"] for word in words)
        has_conversation_term = any(
            word in ["talking", "speaking", "chatting"] for word in words
        )

        # Check for question structure
        is_question = (
            text.endswith("?")
            or has_question_word
            or any(
                tag in ["WP", "WRB"] for word, tag in tagged
            )  # WP = wh-pronoun, WRB = wh-adverb
        )

        # Combine conditions for identity questions
        is_identity_question = is_question and (
            (has_identity_term) or (has_question_word and has_conversation_term)
        )

        if is_identity_question:
            self.logger.info(f"Detected identity question: {text}")

        return is_identity_question

    def store_identity(self, identity: str) -> None:
        """Store personal identity in database and add to recent entities"""
        if not identity:
            return

        try:
            with self.get_connection() as conn:
                now = datetime.now()

                # Store in identity table
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity (id, name, last_updated)
                    VALUES (1, ?, ?)
                    """,
                    (identity, now),
                )

                # Store in memory table instead of entities table
                self.store_entity(identity)  # Use existing store_entity method
                conn.commit()

                # Verify storage
                cur = conn.cursor()
                cur.execute("SELECT name FROM identity WHERE id = 1")
                self.logger.info(f"Verified identity storage: {cur.fetchone()}")
        except sqlite3.Error as e:
            self.logger.error(f"Database error while storing identity: {e}")

    def load_identity(self) -> str | None:
        """Load personal identity from database"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM identity WHERE id = 1")
                result = cur.fetchone()
                if result:
                    self.personal_identity = result[0]
                    self.logger.info(
                        f"Loaded identity from database: {self.personal_identity}"
                    )
                else:
                    self.logger.info("No identity found in database")
                return self.personal_identity
        except sqlite3.Error as e:
            self.logger.error(f"Database error while loading identity: {e}")
            return None

    def pre_send_hook(self, conversation: sm.Conversation):
        last_message = conversation.get_last_message(role="user")
        if not last_message:
            return

        self.logger.info(f"Processing message: {last_message.text}")

        # Check for identity statements FIRST
        identity = self.extract_identity(last_message.text)
        if identity:
            self.logger.info(f"Extracted identity: {identity}")
            self.personal_identity = identity
            self.store_identity(identity)
            conversation.add_message(
                role="assistant", text=f"I'll remember that your name is {identity}."
            )
            return False

        # Handle identity questions
        if self.is_identity_question(last_message.text):
            self.load_identity()  # Reload identity from database
            conversation.add_message(
                role="assistant",
                text=(
                    f"You are {self.personal_identity}."
                    if self.personal_identity
                    else "I don't know your name yet. You can tell me by saying 'I am [your name]' or 'My name is [your name]'."
                ),
            )
            return False

        # Extract and store entities
        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.store_entity(entity)
            self.logger.info(f"Stored entity: {entity}")

        if not entities:
            self.logger.info("No entities found in message")

        # Add context message
        recent_entities = self.retrieve_recent_entities()
        context_message = self.format_context_message(recent_entities)
        if context_message:  # Only add if there's actual context to share
            conversation.add_message(role="user", text=context_message)
            self.logger.info(f"Added context message: {context_message}")


# Replace the example usage code at the bottom with this chat interface:
def main():
    # Create a conversation and add the plugin
    conversation = sm.create_conversation(llm_model="gpt-4", llm_provider="openai")
    plugin = EnhancedContextPlugin()
    conversation.add_plugin(plugin)

    # Add initial context if available
    recent_entities = plugin.retrieve_recent_entities()
    context_message = plugin.format_context_message(recent_entities)
    if context_message:
        conversation.add_message(role="system", text=context_message)
        plugin.logger.info(f"Added initial context message: {context_message}")

    console = Console()
    console.print(
        Panel("[bold green]Chat interface ready![/bold green] Type 'quit' to exit.")
    )
    print("-" * 50)

    try:
        while True:
            # Get user input with colored prompt
            console.print("\n[bold blue]You:[/bold blue] ", end="")
            user_input = input().strip()

            # Check for quit command
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[bold green]Goodbye![/bold green]")
                break

            # Add user message and get response
            conversation.add_message(role="user", text=user_input)
            should_continue = plugin.pre_send_hook(conversation)

            # Only send to LLM if pre_send_hook returns True or None
            if should_continue is not False:
                response = conversation.send()
                console.print(
                    "\n[bold purple]Assistant:[/bold purple]",
                    Text(str(response.text), style="italic"),
                )
            else:
                # Get the last assistant message that was added by the plugin
                response = conversation.get_last_message(role="assistant")
                if response:
                    console.print(
                        "\n[bold purple]Assistant:[/bold purple]",
                        Text(response.text, style="bold green"),
                    )

            console.print(Text("-" * 50, style="dim"))
    except KeyboardInterrupt:
        console.print("\n\n[bold green]Goodbye![/bold green]")
        return


if __name__ == "__main__":
    main()
