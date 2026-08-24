# sok2.py

# --- Steg 3: Retrieval - finn relevant tekst i informasjon.txt ---

with open("informasjon.txt", "r", encoding="utf-8") as fil:
    avsnitt = fil.read().split("\n\n")  # del filen i avsnitt (adskilt med tom linje)

sokeord = "stortinget"
sporsmal = "Hvor mange representanter har Stortinget?"

kontekst = ""
for a in avsnitt:
    if sokeord in a.lower():
        kontekst = a.strip()
        break

if kontekst == "":
    kontekst = "Fant ingen relevant informasjon."

print("=== Retrieval-resultat ===")
print(kontekst)
print()

# --- Steg 4: Kombiner - sett innholdet inn i malen ---

with open("mal.md", "r", encoding="utf-8") as fil:
    mal = fil.read()

ferdig_prompt = mal.replace("[SETT INN RETRIEVAL-RESULTAT HER]", kontekst)
ferdig_prompt = ferdig_prompt.replace("[SETT INN SPØRSMÅL HER]", sporsmal)

# Lagre den ferdige prompten til fil, slik at du kan åpne og kopiere den
with open("ferdig-prompt.md", "w", encoding="utf-8") as fil:
    fil.write(ferdig_prompt)

print("=== Ferdig prompt (lagret i ferdig-prompt.md) ===")
print(ferdig_prompt)