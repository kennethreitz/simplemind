from typing import Literal

from _context import sm
from pydantic import BaseModel

# Note: you should probably be using textblob for this.

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float


print(
    sm.generate_data(
        prompt="Analyze the sentiment of the following text:\n\n'The product arrived late and was broken. Worst purchase ever!'",
        llm_provider="openai",
        llm_model="gpt-4o",
        response_model=SentimentAnalysis,
    )
)
# Output: SentimentAnalysis(sentiment='negative', confidence=0.95)
