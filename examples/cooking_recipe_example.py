from pydantic import BaseModel
import simplemind as sm


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
        output = f"\n=== {self.name.upper()} ===\n\n"
        
        output += "INGREDIENTS:\n"
        for ing in self.ingredients:
            output += f"• {ing.quantity} {ing.unit} {ing.name}\n"
        
        output += "\nINSTRUCTIONS:\n"
        for step in self.instructions:
            output += f"{step.step_number}. {step.instruction}\n"
        
        return output
    

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

