from datetime import datetime, timedelta
import logging
import sqlite3
from typing import List
import re
import os
import contextlib

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
from rich.markdown import Markdown
from rich.status import Status

from concurrent.futures import ThreadPoolExecutor
import random

from docopt import docopt

DB_PATH = "enhanced_context.db"
AVAILABLE_PROVIDERS = ["xai", "openai", "anthropic", "ollama"]

__doc__ = """Enhanced Context Chat Interface

Usage:
    enhanced_context.py [--provider=<provider>]
    enhanced_context.py (-h | --help)

Options:
    -h --help                   Show this screen.
    --provider=<provider>       LLM provider to use (openai/anthropic/xai/ollama)
"""


class EnhancedContextPlugin(sm.BasePlugin):
    model_config = {"extra": "allow"}

    def __init__(self, verbose: bool = False):
        super().__init__()
        # Set up logging
        self.verbose = verbose
        if verbose:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(level=logging.WARNING)
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

        # Download required NLTK data silently
        try:
            with open(os.devnull, "w") as null_out:
                with (
                    contextlib.redirect_stdout(null_out),
                    contextlib.redirect_stderr(null_out),
                ):
                    nltk.download("punkt", quiet=True)
                    nltk.download("averaged_perceptron_tagger", quiet=True)
        except LookupError as e:
            self.logger.error(f"Error downloading NLTK data: {e}")

        # Add LLM personality traits for easter egg
        self.llm_personalities = [
            "You are a wise philosopher who speaks in riddles",
            "You are an excited scientist who loves discovering patterns",
            "You are a detective who analyzes every detail",
            "You are a poet who sees beauty in connections",
            "You are a historian who relates everything to the past",
        ]

        # Add these lines to store the conversation's model and provider
        self.llm_model = None
        self.llm_provider = None

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
            # Modify memory table to include source
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    entity TEXT,
                    source TEXT,  -- 'user' or 'llm'
                    last_mentioned TIMESTAMP,
                    mention_count INTEGER DEFAULT 1,
                    PRIMARY KEY (entity, source)
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

            # Create essence markers table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS essence_markers (
                    marker_type TEXT,
                    marker_text TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (marker_type, marker_text)
                )
            """
            )

    def store_entity(self, entity: str, source: str = "user") -> None:
        """Store or update entity mention with source tracking"""
        try:
            with self.get_connection() as conn:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT INTO memory (entity, source, last_mentioned, mention_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(entity, source) DO UPDATE SET
                        last_mentioned = ?,
                        mention_count = mention_count + 1
                    """,
                    (entity, source, now, now),
                )
                conn.commit()
                self.logger.info(f"Stored {source} entity: {entity}")
        except sqlite3.Error as e:
            self.logger.error(f"Database error while storing entity {entity}: {e}")

    def retrieve_recent_entities(self, days: int = 7) -> List[tuple]:
        """Retrieve recently mentioned entities with frequency and source"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT
                        entity,
                        SUM(mention_count) as total_mentions,
                        GROUP_CONCAT(source || ':' || mention_count) as source_counts
                    FROM memory
                    WHERE last_mentioned >= datetime('now', ?, 'localtime')
                    GROUP BY entity
                    ORDER BY total_mentions DESC, MAX(last_mentioned) DESC
                    LIMIT 5
                    """,
                    (f"-{days} days",),
                )

                entities = []
                for row in cur.fetchall():
                    entity, total_count, source_counts = row
                    source_dict = dict(sc.split(":") for sc in source_counts.split(","))
                    entities.append(
                        (
                            entity,
                            total_count,
                            int(source_dict.get("user", 0)),
                            int(source_dict.get("llm", 0)),
                        )
                    )

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
        """Format context message with source awareness"""
        context_parts = []

        # Add identity context
        if include_identity and self.personal_identity:
            context_parts.append(f"The user's name is {self.personal_identity}.")

        # Add entity context with source awareness
        if entities:
            entity_strings = []
            for entity, total, user_count, llm_count in entities:
                source_info = []
                if user_count > 0:
                    source_info.append(f"{user_count} by user")
                if llm_count > 0:
                    source_info.append(f"{llm_count} by assistant")
                entity_strings.append(f"{entity} (mentioned {', '.join(source_info)})")

            context_parts.append(
                "Recent conversation topics: "
                + (
                    ", ".join(entity_strings[:-1]) + f" and {entity_strings[-1]}"
                    if len(entity_strings) > 1
                    else entity_strings[0]
                )
            )

            # Add guidance for heavily mentioned entities
            heavy_mentions = [(e, t) for e, t, _, _ in entities if t > 3]
            if heavy_mentions:
                context_parts.append(
                    "Note: Be mindful not to overuse "
                    + ", ".join(f"{e}" for e, _ in heavy_mentions)
                    + " as they have been frequently discussed."
                )

        return "\n".join(context_parts)

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

    def is_memory_question(self, text: str) -> bool:
        """Detect questions about memory and recall"""
        text = text.lower().strip()

        # Keywords related to memory and recall
        memory_words = {
            "remember",
            "recall",
            "memory",
            "memories",
            "mentioned",
            "talked about",
            "discussed",
            "tell me about",
            "what do you know",
        }

        return any(word in text for word in memory_words)

    def extract_essence_markers(self, text: str) -> List[tuple[str, str]]:
        """Extract essence markers from text.
        Returns list of tuples (marker_type, marker_text)"""

        # Common patterns for essence markers
        patterns = {
            "value": [
                r"I (?:really )?(?:believe|think) (?:that )?(.+)",
                r"(?:It's|Its) important (?:to me )?that (.+)",
                r"I value (.+)",
                r"(?:The )?most important (?:thing|aspect) (?:to me )?is (.+)",
            ],
            "identity": [
                r"I am(?: a| an)? (.+)",
                r"I consider myself(?: a| an)? (.+)",
                r"I identify as(?: a| an)? (.+)",
            ],
            "preference": [
                r"I (?:really )?(?:like|love|enjoy|prefer) (.+)",
                r"I can't stand (.+)",
                r"I hate (.+)",
                r"I always (.+)",
                r"I never (.+)",
            ],
            "emotion": [
                r"I feel (.+)",
                r"I'm feeling (.+)",
                r"(?:It|That) makes me feel (.+)",
            ],
        }

        markers = []

        # Process with spaCy for better sentence splitting
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip().lower()

            # Check each pattern type
            for marker_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.finditer(pattern, sent_text, re.IGNORECASE)
                    for match in matches:
                        marker_text = match.group(1).strip()
                        # Filter out very short or common phrases
                        if len(marker_text) > 3 and not any(
                            w in marker_text for w in ["um", "uh", "like"]
                        ):
                            markers.append((marker_type, marker_text))

        return markers

    def store_essence_marker(self, marker_type: str, marker_text: str) -> None:
        """Store essence marker in database"""
        try:
            with self.get_connection() as conn:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO essence_markers (marker_type, marker_text, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (marker_type, marker_text, now),
                )
                conn.commit()
                self.logger.info(
                    f"Stored essence marker: {marker_type} - {marker_text}"
                )
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing essence marker: {e}")

    def retrieve_essence_markers(self, days: int = 30) -> List[tuple[str, str]]:
        """Retrieve recent essence markers"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT marker_type, marker_text
                    FROM essence_markers
                    WHERE timestamp >= datetime('now', ?, 'localtime')
                    ORDER BY timestamp DESC
                """,
                    (f"-{days} days",),
                )

                markers = cur.fetchall()
                self.logger.info(f"Retrieved essence markers: {markers}")
                return markers
        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving essence markers: {e}")
            return []

    def format_context_message(
        self, entities: List[tuple], include_identity: bool = True
    ) -> str:
        """Format context message with essence markers"""
        context_parts = []

        # Add identity context
        if include_identity and self.personal_identity:
            context_parts.append(f"The user's name is {self.personal_identity}.")

        # Add essence markers
        essence_markers = self.retrieve_essence_markers()
        if essence_markers:
            markers_by_type = {}
            for marker_type, marker_text in essence_markers:
                if marker_type not in markers_by_type:
                    markers_by_type[marker_type] = []
                markers_by_type[marker_type].append(marker_text)

            context_parts.append("User characteristics:")
            for marker_type, markers in markers_by_type.items():
                context_parts.append(f"- {marker_type.title()}: {', '.join(markers)}")

        # Add entity context - Fixed tuple unpacking
        if entities:
            entity_strings = [
                f"{entity} (mentioned {total} {'times' if total > 1 else 'time'})"
                for entity, total, user_count, llm_count in entities
            ]
            context_parts.append(
                "Recent conversation topics: "
                + (
                    ", ".join(entity_strings[:-1]) + f" and {entity_strings[-1]}"
                    if len(entity_strings) > 1
                    else entity_strings[0]
                )
            )

        return "\n".join(context_parts)

    def pre_send_hook(self, conversation: sm.Conversation):
        # Add these lines at the start of pre_send_hook
        self.llm_model = conversation.llm_model
        self.llm_provider = conversation.llm_provider

        last_message = conversation.get_last_message(role="user")
        if not last_message:
            return

        self.logger.info(f"Processing user message: {last_message.text}")

        # Extract and store entities with user source
        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.store_entity(entity, source="user")

        # Extract and store essence markers
        essence_markers = self.extract_essence_markers(last_message.text)
        for marker_type, marker_text in essence_markers:
            self.store_essence_marker(marker_type, marker_text)
            self.logger.info(f"Found essence marker: {marker_type} - {marker_text}")

        # Add context message
        recent_entities = self.retrieve_recent_entities(days=30)
        context_message = self.format_context_message(recent_entities)
        if context_message:
            conversation.add_message(role="user", text=context_message)
            self.logger.info(f"Added context message: {context_message}")

        return True

    def post_response_hook(self, conversation: sm.Conversation):
        """Process the LLM's response to extract and store entities"""
        last_response = conversation.get_last_message(role="assistant")
        if not last_response:
            return

        self.logger.info(f"Processing assistant response: {last_response.text}")

        # Extract and store entities from the LLM's response with llm source
        entities = self.extract_entities(last_response.text)
        for entity in entities:
            self.store_entity(entity, source="llm")

        return True

    def simulate_llm_conversation(self, context: str, num_turns: int = 3) -> str:
        """Simulate a conversation between multiple LLM personalities about the context"""
        conversation_log = []

        def get_response(personality: str, previous_messages: str) -> str:
            prompt = (
                f"{personality}. You are participating in a brief group discussion "
                f"about the following context:\n{context}\n\n"
                f"Previous messages:\n{previous_messages}\n\n"
                "Provide a short, focused response (1-2 sentences) that builds on "
                "the discussion. Be creative but stay on topic."
            )

            temp_conv = sm.create_conversation(
                llm_model=self.llm_model, llm_provider=self.llm_provider
            )
            temp_conv.add_message(role="user", text=prompt)
            response = temp_conv.send()
            return response.text.strip()

        # Select random personalities for this conversation
        selected_personalities = random.sample(
            self.llm_personalities, min(num_turns, len(self.llm_personalities))
        )

        with ThreadPoolExecutor() as executor:
            for i, personality in enumerate(selected_personalities, 1):
                previous = "\n".join(conversation_log)
                response = get_response(personality, previous)
                conversation_log.append(f"Speaker {i}: {response}")

        return "\n\n".join(conversation_log)


# Replace the example usage code at the bottom with this chat interface:
def main():
    # Parse arguments
    args = docopt(__doc__)
    console = Console()

    # Use command line provider if specified, otherwise use picker
    if args["--provider"]:
        provider = args["--provider"].lower()
        model = None
    else:
        provider = None
        model = None

    # Create a conversation and add the plugin
    conversation = sm.create_conversation(llm_model=model, llm_provider=provider)
    plugin = EnhancedContextPlugin(verbose=False)  # Set verbose here
    conversation.add_plugin(plugin)

    # Add initial context if available
    recent_entities = plugin.retrieve_recent_entities()
    context_message = plugin.format_context_message(recent_entities)
    if context_message:
        conversation.add_message(role="user", text=context_message)
        plugin.logger.info(f"Added initial context message: {context_message}")

    console = Console()
    md = """# Enhanced Context Chat Interface
Type 'quit' to exit. Type 'go go go' for a special surprise!

---"""
    console.print(Markdown(md))

    try:
        while True:
            console.print(Markdown("**You:**"), end=" ")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print(Markdown("**Goodbye!**"))
                break

            # Easter egg handling
            if user_input.lower() == "go go go":
                console.print(Markdown("## 🎉 Multi-LLM Discussion Initiated!"))
                recent_entities = plugin.retrieve_recent_entities()
                context = plugin.format_context_message(recent_entities)
                conversation_result = plugin.simulate_llm_conversation(context)

                console.print(
                    Markdown(
                        f"""### Discussion Results
*{conversation_result}*

---"""
                    )
                )
                continue

            # Regular conversation handling
            conversation.add_message(role="user", text=user_input)
            should_continue = plugin.pre_send_hook(conversation)

            if should_continue is not False:
                with Status("[bold blue]Thinking...", spinner="dots") as status:
                    response = conversation.send()
                    plugin.post_response_hook(conversation)
                console.print(
                    Markdown(
                        f"""**Assistant:** *{response.text}*

---"""
                    )
                )
            else:
                response = conversation.get_last_message(role="assistant")
                if response:
                    console.print(
                        Markdown(
                            f"""**Assistant:** *{response.text}*

---"""
                        )
                    )

    except KeyboardInterrupt:
        console.print(Markdown("\n**Goodbye!**"))
        return


if __name__ == "__main__":
    main()
