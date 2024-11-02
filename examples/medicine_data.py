from pydantic import BaseModel
from _context import simplemind as sm


class SideEffect(BaseModel):
    effect: str
    severity: str  # mild, moderate, severe
    frequency: str  # common, uncommon, rare


class Medication(BaseModel):
    brand_name: str
    generic_name: str
    drug_class: str
    half_life: str
    common_uses: list[str]
    side_effects: list[SideEffect]
    typical_dosage: str
    warnings: list[str]


class MedicationList(BaseModel):
    root: list[Medication]


# Create a session with your preferred model
session = sm.Session(llm_provider="openai", llm_model="gpt-4o-mini")


# Update the prompt to use an f-string with a parameter
def get_medication_prompt(medications: list[str]) -> str:
    return f"""
Provide detailed medical information about {', '.join(medications)}.
Include their generic names, drug classes, half-lives, common uses, side effects (with severity and frequency),
typical dosages, and important warnings.
Return the information as separate medication entries.
"""


# Example usage
medications_to_lookup = ["Abilify (aripiprazole)", "Trileptal (oxcarbazepine)"]
prompt = get_medication_prompt(medications_to_lookup)

# Generate structured data for medications
medications = session.generate_data(prompt=prompt, response_model=MedicationList)

# Print the structured information
for med in medications.root:
    print(f"\n=== {med.brand_name} ===")
    print(f"Generic Name: {med.generic_name}")
    print(f"Drug Class: {med.drug_class}")
    print(f"Half Life: {med.half_life}")
    print("\nCommon Uses:")
    for use in med.common_uses:
        print(f"- {use}")
    print("\nSide Effects:")
    for effect in med.side_effects:
        print(f"- {effect.effect} ({effect.severity}, {effect.frequency})")
    print(f"\nTypical Dosage: {med.typical_dosage}")
    print("\nWarnings:")
    for warning in med.warnings:
        print(f"- {warning}")
