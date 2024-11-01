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