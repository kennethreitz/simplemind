from typing import Optional, Type
from pydantic import BaseModel

from simplemind import Session

# Example usage:
if __name__ == "__main__":
    # Create a session with default settings
    session = Session(llm_provider="openai", llm_model="gpt-4o-mini")

    # Use the session for text generation
    response = session.generate_text("Tell me about the future of AI")

    # Create a conversation using session defaults
    conversation = session.create_conversation()

    # Generate structured data with Pydantic
    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        instructions: list[str]

    recipe = session.generate_data(
        "Give me a recipe for chocolate chip cookies", response_model=Recipe
    )

    print(locals())
