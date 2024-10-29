from typing import List

from pydantic import BaseModel

from _context import sm


class Movie(BaseModel):
    title: str
    year: int

class MovieCharecter(BaseModel):
    name: str
    actor: str

class MovieQuote(BaseModel):
    quote: str
    movie: Movie
    charecter: MovieCharecter

class QuotesList(BaseModel):
    quotes: List[MovieQuote]
    theme: str


quotes = sm.generate_data(llm_provider="openai", llm_model="gpt-4o-mini", prompt="Generate 20 quotes from famous movies", response_model=QuotesList)

for quote in quotes.quotes:
    print(f"{quote.charecter.name} from {quote.movie.title} ({quote.movie.year}): {quote.quote!r}")
