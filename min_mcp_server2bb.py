
from fastmcp import FastMCP
 
mcp = FastMCP("Plattform-Assistent")
 
 
@mcp.resource("plattform://status")
def hent_plattform_status() -> str:
    """Returnerer generell status for plattform AA-203."""
    return "Plattform AA-203: Normal drift. Dekk 2 er operativt. Verksted er tilgjengelig."
 
 
@mcp.tool()
def sjekk_lokasjon(navn: str) -> str:
    """Bruk dette verktøyet for å finne ut hvor et rom, system eller utstyr
    (f.eks. brannvarsler, kompressor, drivstofftank) befinner seg på plattform AA-203."""
    navn_lav = navn.lower()
 
    database = {
        # Rom
        "verksted": "Verkstedet ligger på Dekk 2, rett ved siden av Adgangskontrollen og Pallelagringen.",
        "kontrollrom": "Kontrollrommet ligger øverst i midten på Dekk 2, direkte over Pallelagringen.",
        "pumperom": "Pumperom (P-304) ligger på høyre side av Dekk 2 og huser pumper P-304A og P-304B.",
 
        # Utstyr / systemer
        "kompressor": "Kompressor-stasjonen (K-102) er plassert nede til venstre på Dekk 2.",
        "brannvarsler": "Brannvarslere er plassert i alle rom på Dekk 2, med hovedpanel i Kontrollrommet.",
        "drivstofftank": "Drivstofftanken (T-201) befinner seg på Dekk 1, under Pumperom P-304.",
        "generator": "Nødgeneratoren står på Dekk 1, nær helikopterdekket.",
    }
 
    treff = [beskrivelse for nokkel, beskrivelse in database.items() if nokkel in navn_lav]
 
    if treff:
        return "\n".join(treff)
 
    return f"Fant ingen lokasjon for '{navn}'. Sjekk om navnet er skrevet riktig, eller om det finnes flere treff."
 
 
if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)