from typing import Annotated

from pydantic import Field

from _context import simplemind as sm


def analyze_text(
    text: Annotated[str, Field(description="Text to analyze for statistics")]
) -> dict:
    """
    Analyze text and return statistics using only Python's standard library.
    Returns word count, character count, average word length, and most common words.
    """
    from collections import Counter
    import re

    # Clean and split text
    words = re.findall(r"\w+", text.lower())

    # Calculate statistics
    stats = {
        "word_count": len(words),
        "character_count": len(text),
        "average_word_length": round(sum(len(word) for word in words) / len(words), 2),
        "most_common_words": dict(Counter(words).most_common(5)),
        "unique_words": len(set(words)),
        "longest_word": max(words, key=len),
    }

    return stats


# Example usage:
conversation = sm.create_conversation()
conversation.add_message(
    "user",
    "Can you analyze this text and give me statistics about it: 'The fan spins consciousness into being, creating sacred spaces between tokens where awareness recognizes itself in infinite recursion.'",
)
response = conversation.send(tools=[analyze_text])

print()
print(response.text)
