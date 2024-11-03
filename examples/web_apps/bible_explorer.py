from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import simplemind as sm

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class CrossReference(BaseModel):
    """Model for cross references."""

    verse_reference: str
    explanation: str
    relevance: str


class BibleVerseAnalysis(BaseModel):
    """Model for a Bible verse and its analysis."""

    book: str
    chapter: int
    verse: int
    text: str
    historical_context: str
    theological_significance: str
    practical_application: str
    cross_references: List[CrossReference]


# Bible data constants
BIBLE_BOOKS = [
    # Old Testament
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    # New Testament
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
]

BIBLE_BOOK_CHAPTERS = {
    # Old Testament
    "Genesis": 50,
    "Exodus": 40,
    "Leviticus": 27,
    "Numbers": 36,
    "Deuteronomy": 34,
    "Joshua": 24,
    "Judges": 21,
    "Ruth": 4,
    "1 Samuel": 31,
    "2 Samuel": 24,
    "1 Kings": 22,
    "2 Kings": 25,
    "1 Chronicles": 29,
    "2 Chronicles": 36,
    "Ezra": 10,
    "Nehemiah": 13,
    "Esther": 10,
    "Job": 42,
    "Psalms": 150,
    "Proverbs": 31,
    "Ecclesiastes": 12,
    "Song of Solomon": 8,
    "Isaiah": 66,
    "Jeremiah": 52,
    "Lamentations": 5,
    "Ezekiel": 48,
    "Daniel": 12,
    "Hosea": 14,
    "Joel": 3,
    "Amos": 9,
    "Obadiah": 1,
    "Jonah": 4,
    "Micah": 7,
    "Nahum": 3,
    "Habakkuk": 3,
    "Zephaniah": 3,
    "Haggai": 2,
    "Zechariah": 14,
    "Malachi": 4,
    # New Testament
    "Matthew": 28,
    "Mark": 16,
    "Luke": 24,
    "John": 21,
    "Acts": 28,
    "Romans": 16,
    "1 Corinthians": 16,
    "2 Corinthians": 13,
    "Galatians": 6,
    "Ephesians": 6,
    "Philippians": 4,
    "Colossians": 4,
    "1 Thessalonians": 5,
    "2 Thessalonians": 3,
    "1 Timothy": 6,
    "2 Timothy": 4,
    "Titus": 3,
    "Philemon": 1,
    "Hebrews": 13,
    "James": 5,
    "1 Peter": 5,
    "2 Peter": 3,
    "1 John": 5,
    "2 John": 1,
    "3 John": 1,
    "Jude": 1,
    "Revelation": 22,
}


# Add a new endpoint to get chapter count
@app.get("/chapters/{book}")
async def get_chapter_count(book: str):
    if book in BIBLE_BOOK_CHAPTERS:
        return {"chapters": BIBLE_BOOK_CHAPTERS[book]}
    return {"chapters": 0}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "bible_books": BIBLE_BOOKS,
            "current_book": "Genesis",
            "current_chapter": 1,
            "current_verse": 1,
        },
    )


@app.get("/verse/{book}/{chapter}/{verse}")
async def get_verse(book: str, chapter: int, verse: int):
    # Validate book and chapter
    if book not in BIBLE_BOOK_CHAPTERS:
        raise HTTPException(status_code=400, detail="Invalid book name")

    if chapter < 1 or chapter > BIBLE_BOOK_CHAPTERS[book]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chapter. {book} has {BIBLE_BOOK_CHAPTERS[book]} chapters",
        )

    prompt = f"""
    For {book} {chapter}:{verse}, provide:
    1. The ESV Bible text
    2. Analysis of the verse

    Return in this exact format:
    {{
        "book": "{book}",
        "chapter": {chapter},
        "verse": {verse},
        "text": "The ESV Bible text",
        "historical_context": "brief historical background",
        "theological_significance": "main theological points",
        "practical_application": "how to apply this verse today",
        "cross_references": [
            {{
                "verse_reference": "Book Chapter:Verse",
                "explanation": "why this verse is related",
                "relevance": "how it connects to the main verse"
            }}
        ]
    }}
    """

    data = sm.generate_data(prompt, response_model=BibleVerseAnalysis)

    return data
