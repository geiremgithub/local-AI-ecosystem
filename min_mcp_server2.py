import json
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Plattform-Assistent")

# JSON-filen ligger i samme mappe som dette scriptet
LOKASJONER_FIL = Path(__file__).parent / "lokasjoner.json"


def last_lokasjoner() -> dict:
    """Leser lokasjonsdatabasen fra JSON-filen på nytt for hvert kall,
    slik at endringer i filen slår inn uten at serveren må restartes."""
    with open(LOKASJONER_FIL, "r", encoding="utf-8") as f:
        return json.load(f)


@mcp.resource("plattform://status")
def hent_plattform_status() -> str:
    """Returnerer generell status for plattform AA-203."""
    return "Plattform AA-203: Normal drift. Dekk 2 er operativt. Verksted er tilgjengelig."


@mcp.tool()
def sjekk_lokasjon(navn: str) -> str:
    """Bruk dette verktøyet for å finne ut hvor et rom, system eller utstyr
    (f.eks. brannvarsler, kompressor, drivstofftank, pumpe) befinner seg på plattform AA-203."""
    navn_lav = navn.lower().replace(" ", "-")
    database = last_lokasjoner()

    treff = []
    for nokkel, data in database.items():
        if nokkel in navn_lav or navn_lav in nokkel or navn.lower() in data["navn"].lower():
            treff.append(f"{data['navn']}: {data['beskrivelse']}")

    if treff:
        return "\n".join(treff)

    return f"Fant ingen lokasjon for '{navn}'. Sjekk om navnet er skrevet riktig, eller om det finnes flere treff."


@mcp.tool()
def list_alle_lokasjoner() -> str:
    """Lister opp alle rom, systemer og utstyr som finnes registrert på plattform AA-203."""
    database = last_lokasjoner()
    return "\n".join(f"- {data['navn']}" for data in database.values())


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
