from fastmcp import FastMCP

mcp = FastMCP("Plattform-Assistent")

@mcp.resource("plattform://status")
def hent_plattform_status() -> str:
    """Returnerer generell status for plattform AA-203."""
    return "Plattform AA-203: Normal drift. Dekk 2 er operativt. Verksted er tilgjengelig."

@mcp.tool()
def sjekk_rom_lokasjon(rom_navn: str) -> str:
    """Bruk dette verktøyet for å finne ut hvor et spesifikt rom ligger på plattform AA-203."""
    rom_navn_lav = rom_navn.lower()

    database = {
        "verksted": "Verkstedet ligger på Dekk 2, rett ved siden av Adgangskontrollen og Pallelagringen.",
        "kontrollrom": "Kontrollrommet ligger øverst i midten på Dekk 2, direkte over Pallelagringen.",
        "pumperom": "Pumperom (P-304) ligger på høyre side av Dekk 2 og huser pumper P-304A og P-304B.",
        "kompressor": "Kompressor-stasjonen (K-102) er plassert nede til venstre på Dekk 2.",
    }

    for nokkel, beskrivelse in database.items():
        if nokkel in rom_navn_lav:
            return beskrivelse

    return f"Fant ingen plassering for '{rom_navn}'. Sjekk om navnet er skrevet riktig."

if __name__ == "__main__":
    mcp.run()