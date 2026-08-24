def les_dokument(filnavn: str) -> str:
    with open(filnavn, "r", encoding="utf-8") as fil:
        innhold = fil.read()
    return innhold

# Vi tester funksjonen med en fil som heter test.txt
print(les_dokument("informasjon.txt"))
import os
print("Arbeidsmappen er:", os.getcwd())

