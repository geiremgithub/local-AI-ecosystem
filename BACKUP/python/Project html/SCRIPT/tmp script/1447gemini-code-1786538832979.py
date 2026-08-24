import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# 1. Start med at brukeren stiller et spørsmål
sporsmal = "Hvilke regler gjelder for fravær på skolen?"
print(f"--- [BRUKERFORESPØRSEL] ---\nStilt spørsmål: \"{sporsmal}\"\n")

# 2. Steg 1: Tokenisering av detstilte spørsmålet
# (Simulert tokenisering for demonstrasjon av ord/del-ord)
tokens = ['Hvil', 'ke', 'regler', 'gjelder', 'for', 'fravær', 'på', 'skolen?']
token_ids = [4320, 381, 12845, 9821, 321, 24150, 312, 18230]

print("--- 1. TOKENISERING ---")
print(f"Spørsmålet ble delt opp i {len(tokens)} tokens.")
print(f" - Tokens: {tokens}")
print(f" - Token ID-er: {token_ids}\n")

# 3. Steg 2: Embedding av detstilte spørsmålet
# Vi bruker en ekte modell til å gjøre om spørsmålet til en vektor (her 384 dimensjoner)
modell = SentenceTransformer('all-MiniLM-L6-v2')
sporsmal_vektor = modell.encode(sporsmal)

print("--- 2. EMBEDDING AV SPØRSMÅL ---")
print(f"Modell: all-MiniLM-L6-v2 genererte vektor med {len(sporsmal_vektor)} dimensjoner.")
print(f" - Vektortall (utdrag): {[round(x, 4) for x in sporsmal_vektor[:6]]} ...\n")

# 4. Steg 3: Lete etter dokumenter i databasen basert på detstilte spørsmålet
# Vi definerer en lokal database med dokument-chunks
database = [
    {"kilde": "skoleregler.txt (Avsnitt 3)", "tekst": "Fravær må dokumenteres med legeerklæring ved sykdom over 3 dager."},
    {"kilde": "reglement.txt (Avsnitt 8)", "tekst": "Elever har møteplikt til alle skoletimer og aktiviteter."},
    {"kilde": "Skoler.txt (Avsnitt 12)", "tekst": "Ferie i annen tid enn skolens ferier innvilges sjelden."},
    {"kilde": "skoleregler.txt (Avsnitt 15)", "tekst": "Varslingsgrensen for fravær i videregående skole er 10 prosent."}
]

# Genererer vektorer for databasen
db_tekster = [item["tekst"] for item in database]
db_vektorer = modell.encode(db_tekster)

# Beregner avstand (cosine/euklidsk) fra spørsmålet til hver vektor i databasen for å "lete"
avstander = [np.linalg.norm(sporsmal_vektor - v) for v in db_vektorer]
sorterte_treff = np.argsort(avstander)

print("--- 3. DATABASELETEOPPDRAG OG VEKTORER ---")
print("Søker i databasen etter treff som matcher detstilte spørsmålet...")
for i, idx in enumerate(sorterte_treff):
    treff = database[idx]
    distanse = avstander[idx]
    vektor_utdrag = [round(x, 4) for x in db_vektorer[idx][:6]]
    print(f" Treff [{i+1}] -> Kilde: {treff['kilde']} (Avstand: {distanse:.4f})")
    print(f"   Tekst: \"{treff['tekst']}\"")
    print(f"   Vektortall: {vektor_utdrag} ...")
print()

# 5. Steg 4: Vektorrom-visualisering av det stilte spørsmålet opp mot databasen
print("--- 4. VEKTORROM-VISUALISERING ---")
print("Genererer 2D-plot av vektorrommet basert på søket...")

# Reduserer dimensjonene fra 384 til 2 ved hjelp av PCA for å kunne plotte det
alle_vektorer = np.vstack([sporsmal_vektor, db_vektorer])
pca = PCA(n_components=2)
koordinater_2d = pca.fit_transform(alle_vektorer)

sporsmal_2d = koordinater_2d[0]
db_2d = koordinater_2d[1:]

plt.figure(figsize=(9, 6))
# Plott databasetreffene i vektorrommet
plt.scatter(db_2d[:, 0], db_2d[:, 1], color='dodgerblue', label='Database-chunks', s=120)
for i, item in enumerate(database):
    plt.annotate(item['kilde'], (db_2d[i, 0], db_2d[i, 1]), fontsize=9, xytext=(6, 6), textcoords='offset points')

# Plott det stilte spørsmålet i det samme vektorrommet
plt.scatter(sporsmal_2d[0], sporsmal_2d[1], color='crimson', label='Stilt Spørsmål', s=180, marker='*')
plt.annotate("Spørsmål", (sporsmal_2d[0], sporsmal_2d[1]), fontsize=10, fontweight='bold', color='crimson', xytext=(6, 6), textcoords='offset points')

plt.title(f"Vektorrom-visualisering for forespørsel: \"{sporsmal}\"")
plt.xlabel("PCA Dimensjon 1")
plt.ylabel("PCA Dimensjon 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

filnavn = "vektorrom_etter_sporsmal.png"
plt.savefig(filnavn, bbox_inches='tight')
plt.close()

print(f" Vektorrom-visualisering lagret som bildefil: {filnavn}")