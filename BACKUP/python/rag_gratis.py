"""
rag_gratis.py
-------------
Et komplett RAG-eksempel (Retrieval-Augmented Generation) som IKKE krever
noen API-nøkkel, kredittkort eller betalt tjeneste.

Programmet gjør de to første RAG-stegene automatisk i kode:

  STEG 1: Les kildedokumentet (informasjon.txt)
  STEG 2: Del det opp i biter / chunks
  STEG 3: Retrieval  - finn den biten som er mest relevant for spørsmålet
  STEG 4: Augmentation - sett biten inn i en promptmal sammen med spørsmålet

Det siste steget (generation) gjøres manuelt og gratis:
  STEG 5: Du kopierer den ferdige prompten (skrevet ut av programmet,
          og lagret i filen ferdig-prompt.md) og limer den inn i en
          GRATIS chat på claude.ai i nettleseren din.

Ingen ekstra biblioteker trengs - bare vanlig Python (ingen "pip install").

Forutsetninger før du kjører:
  - Filene informasjon.txt og mal.md må ligge i samme mappe som dette skriptet
"""


# ---------------------------------------------------------------------------
# STEG 1: Les kildedokumentet
# ---------------------------------------------------------------------------
def les_dokument(filnavn: str) -> str:
    """
    Åpner en tekstfil og returnerer hele innholdet som én lang tekststreng.
    """
    with open(filnavn, "r", encoding="utf-8") as fil:
        innhold = fil.read()
    return innhold


# ---------------------------------------------------------------------------
# STEG 2: Del dokumentet i biter (chunking)
# ---------------------------------------------------------------------------
def del_i_biter(tekst: str) -> list[str]:
    """
    Deler den lange teksten i mindre biter (chunks).

    Her bruker vi en enkel regel: hvert avsnitt (tekst adskilt av en
    tom linje) blir én egen bit. Dette gjør det mulig å søke i og
    hente ut EN bit av gangen, i stedet for å bruke hele dokumentet.
    """
    rå_biter = tekst.split("\n\n")
    biter = [bit.strip() for bit in rå_biter if bit.strip() != ""]
    return biter


# ---------------------------------------------------------------------------
# STEG 3: Retrieval - finn den mest relevante biten
# ---------------------------------------------------------------------------
def finn_relevant_bit(biter: list[str], sporsmal: str) -> str:
    """
    Enkel nøkkelordbasert retrieval.

    Vi teller, for hver bit, hvor mange av ordene fra spørsmålet som
    finnes i biten. Biten med flest treff regnes som mest relevant.

    Dette er en forenklet variant av retrieval. I et ekte RAG-system
    ville man i stedet konvertert både spørsmål og biter til
    tallvektorer (embeddings) og funnet den biten som ligger nærmest
    spørsmålet i "betydning" - ikke bare eksakte ordtreff.
    """
    sporsmal_ord = sporsmal.lower().split()

    beste_bit = None
    beste_poengsum = 0

    for bit in biter:
        bit_lowercase = bit.lower()
        poengsum = sum(1 for ord in sporsmal_ord if ord in bit_lowercase)

        if poengsum > beste_poengsum:
            beste_poengsum = poengsum
            beste_bit = bit

    if beste_bit is None:
        return "Fant ingen relevant informasjon i dokumentet."

    return beste_bit


# ---------------------------------------------------------------------------
# STEG 4: Augmentation - bygg den ferdige prompten fra mal + kontekst
# ---------------------------------------------------------------------------
def bygg_prompt(mal_filnavn: str, kontekst: str, sporsmal: str) -> str:
    """
    Leser inn promptmalen (mal.md) og fyller inn plassholderne med
    den relevante konteksten og spørsmålet.
    """
    with open(mal_filnavn, "r", encoding="utf-8") as fil:
        mal = fil.read()

    ferdig_prompt = mal.replace("[SETT INN RETRIEVAL-RESULTAT HER]", kontekst)
    ferdig_prompt = ferdig_prompt.replace("[SETT INN SPØRSMÅL HER]", sporsmal)

    return ferdig_prompt


# ---------------------------------------------------------------------------
# Hovedprogram - kjører de automatiserte stegene og forbereder deg på steg 5
# ---------------------------------------------------------------------------
def main():
    sporsmal = "Hvor mange representanter har Stortinget?"

    print("=" * 60)
    print("STEG 1+2: Leser og deler opp kildedokumentet")
    print("=" * 60)
    dokument = les_dokument("informasjon.txt")
    biter = del_i_biter(dokument)
    print(f"Dokumentet ble delt i {len(biter)} biter.\n")

    print("=" * 60)
    print("STEG 3: Retrieval")
    print("=" * 60)
    kontekst = finn_relevant_bit(biter, sporsmal)
    print("Fant denne biten som mest relevant:")
    print(kontekst)
    print()

    print("=" * 60)
    print("STEG 4: Augmentation (bygger ferdig prompt)")
    print("=" * 60)
    ferdig_prompt = bygg_prompt("mal.md", kontekst, sporsmal)
    print(ferdig_prompt)

    # Lagre den ferdige prompten til fil, slik at den er lett å åpne og kopiere
    with open("ferdig-prompt.md", "w", encoding="utf-8") as fil:
        fil.write(ferdig_prompt)

    print("=" * 60)
    print("STEG 5: Generation (gjøres manuelt, gratis)")
    print("=" * 60)
    print("Den ferdige prompten er lagret i filen 'ferdig-prompt.md'.")
    print("Åpne den filen, kopier alt innholdet, og lim det inn i en")
    print("GRATIS chat på https://claude.ai for å få et generert svar.")
    print("Ingen API-nøkkel eller betaling er nødvendig for dette steget.")


if __name__ == "__main__":
    main()
