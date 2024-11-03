from pydantic import BaseModel
from _context import simplemind as sm
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class InstructionStep(BaseModel):
    step_number: int
    instruction: str


class RecipeIngredient(BaseModel):
    name: str
    quantity: float
    unit: str


class Recipe(BaseModel):
    name: str
    ingredients: list[RecipeIngredient]
    instructions: list[InstructionStep]

    def __str__(self) -> str:
        console = Console(record=True, file=None)

        # Create formatted title with more emphasis
        title = Text("✨ " + self.name.upper() + " ✨", style="bold blue")

        # Format ingredients with better structure
        ingredients_text = Text("\n📝 INGREDIENTS:\n", style="bold green")
        for ing in self.ingredients:
            # Format numbers to avoid floating decimals when whole numbers
            quantity = int(ing.quantity) if ing.quantity.is_integer() else ing.quantity
            ingredients_text.append(f"  • {quantity} {ing.unit} ", style="bright_white")
            ingredients_text.append(f"{ing.name}\n", style="italic bright_white")

        # Format instructions with better spacing and styling
        instructions_text = Text("\n👩‍🍳 INSTRUCTIONS:\n", style="bold yellow")
        for step in self.instructions:
            instructions_text.append(
                f"\n  {step.step_number}. ", style="bold bright_white"
            )
            instructions_text.append(f"{step.instruction}", style="bright_white")

        # Combine all text
        full_text = Text.assemble(
            ingredients_text, instructions_text, "\n"
        )  # Added extra newline

        # Create panel with enhanced styling
        panel = Panel(
            full_text,
            title=title,
            border_style="blue",
            padding=(1, 2),  # Add padding (vertical, horizontal)
            expand=False,  # Don't expand to full terminal width
            title_align="center",
        )

        # Render the panel to string without printing
        with console.capture() as capture:
            console.print(panel)
        return capture.get()


recipe = sm.generate_data(
    "Write a recipe for chocolate chip cookies",
    llm_model="gpt-4o-mini",
    llm_provider="openai",
    response_model=Recipe,
)

print(recipe)
# Expected output is something like this:
#
# === CHOCOLATE CHIP COOKIES ===
#
# INGREDIENTS:
# • 2.25 cups all-purpose flour
# • 1.0 teaspoon baking soda
# • 0.5 teaspoon salt
# • 1.0 cup unsalted butter
# • 0.75 cup sugar
# • 0.75 cup brown sugar
# • 1.0 teaspoon vanilla extract
# • 2.0 large eggs
# • 2.0 cups semi-sweet chocolate chips
#
# INSTRUCTIONS:
# 1. Preheat your oven to 350°F (175°C).
# 2. In a small bowl, combine flour, baking soda, and salt; set aside.
# 3. In a large bowl, cream together the butter, sugar, and brown sugar until smooth.
# 4. Beat in the vanilla extract and eggs, one at a time.
# 5. Gradually blend in the flour mixture until just combined.
# 6. Stir in the chocolate chips.
# 7. Drop by rounded tablespoon onto ungreased cookie sheets.
# 8. Bake for 9 to 11 minutes, or until edges are golden.
# 9. Let cool on the cookie sheet for a few minutes before transferring to wire racks to cool completely.
