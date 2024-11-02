from _context import simplemind as sm
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

gpt_4o_mini = sm.Session(llm_provider="openai")
claude_sonnet = sm.Session(llm_provider="anthropic")


class BibleVerse(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str
    translation: str


class BiblePassage(BaseModel):
    book: str
    chapter: int
    verses: list[BibleVerse]
    translation: str


class CrossReference(BaseModel):
    passage: BiblePassage
    notes: list[str]
    origin_verse: BibleVerse
    ai_perspective: str
    anthropic_perspective: str


def get_passage(book: str, chapter: int, translation: str = "ESV") -> BiblePassage:
    passage = gpt_4o_mini.generate_data(
        prompt=f"""Return {book} chapter {chapter} from the {translation} translation.
Format each verse as plain text without any special characters or formatting.
For example:
- "Love is patient, love is kind."
- "It does not envy, it does not boast"

Return only the biblical text, formatted as a BiblePassage object.""",
        response_model=BiblePassage,
        max_tokens=8000,
    )
    return passage


def get_cross_reference(passage: BiblePassage) -> CrossReference:
    verses_text = "\n".join([f"Verse {v.verse}: {v.text}" for v in passage.verses])

    # Get main cross-reference from OpenAI
    ref = gpt_4o_mini.generate_data(
        prompt=f"""Find a thematically related Bible passage that connects with this text:
{verses_text}

Return a CrossReference object with:
1. A related passage (using plain text without special characters)
2. A list of clear, specific notes explaining the thematic connections
3. The original passage included
4. An AI perspective that provides a thoughtful, modern interpretation of how these passages relate to contemporary life and universal human experiences""",
        response_model=CrossReference,
    )

    # Get Anthropic's perspective separately
    anthropic_insight = claude_sonnet.generate_text(
        prompt=f"""Analyze these biblical passages from a philosophical and ethical perspective:

Original passage:
{verses_text}

Cross-reference passage:
{' '.join([f'Verse {v.verse}: {v.text}' for v in ref.passage.verses])}

Provide a thoughtful analysis focusing on the philosophical and ethical implications of these passages, drawing from your training in ethics and philosophy.
Return your response as a plain string.""",
    )

    # Add Anthropic's perspective to the reference object
    ref.anthropic_perspective = anthropic_insight
    return ref


def pretty_print_reference(ref: CrossReference):
    # Create origin passage panel
    origin_text = Text()
    origin_text.append(
        f"{ref.origin_verse.book} {ref.origin_verse.chapter}\n",
        style="bold blue",
    )
    origin_text.append(f"{ref.origin_verse.verse}. ", style="blue")
    origin_text.append(f"{ref.origin_verse.text}\n", style="italic")
    origin_text.append(f"\n({ref.origin_verse.translation})", style="dim")

    origin_panel = Panel(origin_text, title="Original Passage", border_style="blue")

    # Create cross reference panel
    ref_text = Text()
    ref_text.append(
        f"{ref.passage.book} {ref.passage.chapter}\n",
        style="bold green",
    )
    for verse in ref.passage.verses:
        ref_text.append(f"{verse.verse}. ", style="green")
        ref_text.append(f"{verse.text}\n", style="italic")
    ref_text.append(f"\n({ref.passage.translation})", style="dim")

    ref_panel = Panel(ref_text, title="Cross Reference", border_style="green")

    # Create notes panel with bullet points
    notes_text = Text()
    for note in ref.notes:
        notes_text.append("• ", style="yellow")
        notes_text.append(f"{note}\n")

    notes_panel = Panel(notes_text, title="Thematic Connections", border_style="yellow")

    # Add new AI perspective panel
    ai_text = Text()
    ai_text.append(ref.ai_perspective)

    ai_panel = Panel(ai_text, title="AI Perspective", border_style="magenta")

    # Print all panels
    console.print(origin_panel)
    console.print(ref_panel)
    console.print(notes_panel)
    console.print(ai_panel)


if __name__ == "__main__":
    # Get 1 Corinthians 13 (The Love Chapter)
    passage = get_passage("1 Corinthians", 13)
    ref = get_cross_reference(passage)
    pretty_print_reference(ref)
