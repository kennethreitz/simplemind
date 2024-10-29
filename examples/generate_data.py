from typing import List, Iterator

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


def gen_quotes(n=10) -> Iterator[MovieQuote]:
    """Generate a list of quotes from famous movies."""

    for q in sm.generate_data(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        prompt=f"Generate {n} quotes from famous movies",
        response_model=QuotesList,
    ).quotes:
        yield q


if __name__ == "__main__":
    for quote in gen_quotes(n=20):
        print(
            f"{quote.charecter.name} from {quote.movie.title} ({quote.movie.year}): {quote.quote!r}"
        )
