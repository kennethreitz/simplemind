from _context import simplemind as sm
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

claude = sm.Session(llm_provider="anthropic")


class BibleVerse(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str
    translation: str


class CrossReference(BaseModel):
    verse: BibleVerse
    notes: str
    origin_verse: BibleVerse


def get_random_bible_verse() -> BibleVerse:
    verse = claude.generate_data(
        prompt="Get a random Bible verse", response_model=BibleVerse
    )

    return verse


def get_cross_reference(verse: BibleVerse) -> CrossReference:
    ref = claude.generate_data(
        prompt=f"Get a cross reference for the verse: {verse.text}",
        response_model=CrossReference,
    )

    return ref


def pretty_print_reference(ref: CrossReference):
    # Create origin verse panel
    origin_text = Text()
    origin_text.append(
        f"{ref.origin_verse.book} {ref.origin_verse.chapter}:{ref.origin_verse.verse}\n",
        style="bold blue",
    )
    origin_text.append(f"{ref.origin_verse.text}\n", style="italic")
    origin_text.append(f"({ref.origin_verse.translation})", style="dim")

    origin_panel = Panel(origin_text, title="Original Verse", border_style="blue")

    # Create cross reference panel
    ref_text = Text()
    ref_text.append(
        f"{ref.verse.book} {ref.verse.chapter}:{ref.verse.verse}\n", style="bold green"
    )
    ref_text.append(f"{ref.verse.text}\n", style="italic")
    ref_text.append(f"({ref.verse.translation})", style="dim")

    ref_panel = Panel(ref_text, title="Cross Reference", border_style="green")

    # Create notes panel
    notes_panel = Panel(ref.notes, title="Connection Notes", border_style="yellow")

    # Print all panels
    console.print(origin_panel)
    console.print(ref_panel)
    console.print(notes_panel)


if __name__ == "__main__":
    verse = get_random_bible_verse()
    ref = get_cross_reference(verse)
    pretty_print_reference(ref)
