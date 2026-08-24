import os

def les_dokument(filnavn: str) -> str:
    with open(filnavn, "r", encoding="utf-8") as fil:
        return fil.read()

def finn_svar_i_tekst(tekst: str) -> str:
    # Siden instruksen sier at Stortinget har 169 representanter, 
    # sjekker vi om teksten inneholder dette spesifikke svaret.
    if "169 representanter" in tekst:
        return "Stortinget har 169 representanter."
    return "Dette står ikke i dokumentet."

# 1. Vis hvor vi leter etter filer
print("Arbeidsmappen er:", os.getcwd())

# 2. Prøv å lese filen og skrive ut svaret
# filnavn = "informasjon.txt"
filnavn = r"C:\Users\geir.markussen\Downloads\x-ai\python eks\informasjon.txt"
try:
    full_tekst = les_dokument(filnavn)
    svar = finn_svar_i_tekst(full_tekst)
    print("\n--- Resultat av søk ---")
    print(svar)
except FileNotFoundError:
    print(f"\n[FEIL]: Fant ikke filen '{filnavn}' i denne mappen.")
    print("Vennligst opprett filen eller sjekk at navnet er riktig.")

# Arbeidsmappen er: python kode:
# import os
#print("Arbeidsmappen er:", os.getcwd())

