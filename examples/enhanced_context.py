from datetime import datetime

import spacy
import sqlite3

from _context import simplemind as sm


class EnhancedContextPlugin(sm.BasePlugin):
    def __init__(self):
        super().__init__()
        # Initialize NLP model and memory database
        object.__setattr__(self, "nlp", spacy.load("en_core_web_sm"))
        object.__setattr__(self, "conn", sqlite3.connect(":memory:"))

        self.init_db()

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
        print(f"Storing entity in memory: {entity}")
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memory (entity, last_mentioned)
                VALUES (?, ?)
            """,
                (entity, datetime.now()),
            )

    def retrieve_recent_entities(self):
        # Retrieve entities mentioned in the last 7 days
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT entity FROM memory
            WHERE last_mentioned >= datetime('now', '-7 days')
        """
        )
        return [row[0] for row in cur.fetchall()]

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
            # Extract entities and store in memory
            entities = self.extract_entities(last_message.text)

            print(f"Extracted entities: {entities}")
            for entity in entities:
                self.store_entity(entity)

            # Retrieve recent entities for context
            recent_entities = self.retrieve_recent_entities()

            if recent_entities:
                print(f"Recent entities found: {recent_entities}")

                context_message = f"Here are some topics recently discussed: {', '.join(recent_entities)}. Feel free to bring them up if relevant."
                conversation.add_message(role="system", text=context_message)


# Create a conversation and add the plugin
conversation = sm.create_conversation(llm_model="gpt-4o-mini", llm_provider="openai")
conversation.add_plugin(EnhancedContextPlugin())


# Replace the single message test with an interactive chat loop
def chat():
    print("Welcome to the enhanced context chat! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        conversation.add_message(role="user", text=user_input)
        response = conversation.send()
        print(f"\nAssistant: {response.text!r}")


if __name__ == "__main__":
    chat()
