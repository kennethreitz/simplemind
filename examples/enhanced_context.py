from datetime import datetime
import logging

import spacy
import sqlite3

from _context import simplemind as sm

DB_PATH = "enhanced_context.db"


class EnhancedContextPlugin(sm.BasePlugin):
    # Add model configuration to allow arbitrary attributes
    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__()  # Don't forget to call parent's __init__
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Initialize NLP model and memory database
        self.nlp = spacy.load("en_core_web_sm")
        self.conn = sqlite3.connect(DB_PATH)
        self.init_db()
        self.logger.info(f"EnhancedContextPlugin initialized with database: {DB_PATH}")

    def init_db(self):
        # Create a table to store entities and their last mention time
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    entity TEXT PRIMARY KEY,
                    last_mentioned TIMESTAMP
                )
            """
            )

    def store_entity(self, entity):
        # Store or update entity mention time
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memory (entity, last_mentioned)
                VALUES (?, ?)
            """,
                (entity, datetime.now()),
            )
        self.logger.info(f"Stored entity: {entity}")

    def retrieve_recent_entities(self):
        # Retrieve entities mentioned in the last 7 days
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT entity FROM memory
            WHERE last_mentioned >= datetime('now', '-7 days')
        """
        )
        entities = [row[0] for row in cur.fetchall()]
        self.logger.info(f"Retrieved recent entities: {entities}")
        return entities

    def extract_entities(self, text):
        # Extract entities (people, places, organizations) from text
        doc = self.nlp(text)
        return [
            ent.text
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "NORP"}
        ]

    def pre_send_hook(self, conversation: sm.Conversation):
        # Process the latest user message
        last_message = conversation.get_last_message(role="user")
        if last_message:
            self.logger.info(f"Processing message: {last_message.text}")

            # Extract entities and store in memory
            entities = self.extract_entities(last_message.text)
            if entities:
                self.logger.info(f"Extracted entities: {entities}")
                for entity in entities:
                    self.store_entity(entity)
            else:
                self.logger.info("No entities found in message")

            # Retrieve recent entities for context
            recent_entities = self.retrieve_recent_entities()
            if recent_entities:
                context_message = f"Here are some topics recently discussed: {', '.join(recent_entities)}. Feel free to bring them up if relevant."
                conversation.add_message(role="system", text=context_message)
                self.logger.info(
                    f"Added context message with entities: {recent_entities}"
                )


# Replace the example usage code at the bottom with this chat interface:
def main():
    # Create a conversation and add the plugin
    conversation = sm.create_conversation(llm_model="gpt-4", llm_provider="openai")
    conversation.add_plugin(EnhancedContextPlugin())

    print("Chat interface ready! Type 'quit' to exit.")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for quit command
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        # Add user message and get response
        conversation.add_message(role="user", text=user_input)
        response = conversation.send()

        # Print assistant's response
        print("\nAssistant:", response)
        print("-" * 50)


if __name__ == "__main__":
    main()
