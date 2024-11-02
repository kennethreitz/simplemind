from pydantic import BaseModel
from _context import simplemind as sm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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

# Create a Rich console
console = Console()

# Replace the print section with Rich formatting
for med in medications.root:
    # Create a table for the medication details
    table = Table(show_header=False, box=None)
    table.add_row("[bold cyan]Generic Name:[/]", med.generic_name)
    table.add_row("[bold cyan]Drug Class:[/]", med.drug_class)
    table.add_row("[bold cyan]Half Life:[/]", med.half_life)

    # Create a nested table for common uses
    uses_table = Table(show_header=False, box=None, padding=(0, 2))
    for use in med.common_uses:
        uses_table.add_row("•", use)

    # Create a nested table for side effects
    effects_table = Table(show_header=False, box=None, padding=(0, 2))
    for effect in med.side_effects:
        severity_color = {"mild": "green", "moderate": "yellow", "severe": "red"}.get(
            effect.severity.lower(), "white"
        )
        effects_table.add_row(
            "•",
            effect.effect,
            f"[{severity_color}]{effect.severity}[/]",
            f"({effect.frequency})",
        )

    # Create a nested table for warnings
    warnings_table = Table(show_header=False, box=None, padding=(0, 2))
    for warning in med.warnings:
        warnings_table.add_row("•", f"[red]{warning}[/]")

    # Add the nested tables to the main table
    table.add_row("[bold cyan]Common Uses:[/]", uses_table)
    table.add_row("[bold cyan]Side Effects:[/]", effects_table)
    table.add_row("[bold cyan]Typical Dosage:[/]", med.typical_dosage)
    table.add_row("[bold cyan]Warnings:[/]", warnings_table)

    # Create and print a panel for each medication
    console.print(
        Panel(table, title=f"[bold blue]{med.brand_name}[/]", border_style="blue")
    )
    console.print()  # Add a blank line between medications
