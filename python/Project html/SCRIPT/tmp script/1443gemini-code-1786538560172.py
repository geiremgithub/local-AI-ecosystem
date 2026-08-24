import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# 1. Simulerer en enkel tokenizer og modell
print("--- [RAG PROSESSSPESIFIKASJON] ---")
sporsmal = "Når starter skolen?"

# Simulert tokenisering (for visningens skyld)
tokens = ['N', 'år', 'starter', 'sk', 'olen']
token_ids = [77, 17900, 26697, 1901, 17205]

print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(tokens)} tokens.")
print(f"   - Tokens: {tokens}")
print(f"   - Token ID-er: {token_ids}")

# 2. Embedding av spørsmålet ved bruk av en ekte modell
model = SentenceTransformer('all-MiniLM-L6-v2')
sporsmal_vektor = model.encode(sporsmal)

print(f"\n2. Embedding (Spørsmål-vektor): Totalt {len(sporsmal_vektor)} dimensjoner.")
print(f"   - Vektortall [format]: {[round(x, 4) for x in sporsmal_vektor[:8]]} ...")

# 3. Simulerer database med dokument-chunks og deres vektorer
db_kilder = [
    {"kilde": "Skoler.txt (Chunk 1)", "tekst": "Skoleåret starter midt i august."},
    {"kilde": "skoleregler.txt (Chunk 11)", "tekst": "Eleven må møte presis ved skolestart."},
    {"kilde": "Skoler.txt (Chunk 12)", "tekst": "Første skoledag er en tirsdag."},
    {"kilde": "skoleregler.txt (Chunk 18)", "tekst": "Søknad om fri leveres rektor."},
    {"kilde": "Skoler.txt (Chunk 19)", "tekst": "Ferier og fridager følger kommunale planer."}
]

# Genererer ekte vektorer for databasetreffene basert på teksten
db_vektorer = [model.encode(item["tekst"]) for item in db_kilder]

print("\n3. Databasesøk...")
print("   - Topp treff fra databasen og deres vektorer:")

# Beregn avstander (euklidsk avstand for eksempel)
avstander = [np.linalg.norm(sporsmal_vektor - v) for v in db_vektorer]
sorterte_indekser = np.argsort(avstander)

for i, idx in enumerate(sorterte_indekser):
    kilde_info = db_kilder[idx]
    dist = avstander[idx]
    vektor_eksempel = [round(x, 4) for x in db_vektorer[idx][:8]]
    print(f"   [{i}] Kilde: {kilde_info['kilde']} | Avstand: {dist:.4f}")
    print(f"       Vektortall: {vektor_eksempel} ...")

# 4. Vektorrom-visualisering (Dimensjonsreduksjon til 2D for plotting)
print("\n[Grafikk] Genererer vektorrom-visualisering...")

from sklearn.decomposition import PCA
# Samler alle vektorer (spørsmål + database) for PCA-reduksjon
alle_vektorer = np.vstack([sporsmal_vektor, np.array(db_vektorer)])
pca = PCA(n_components=2)
koordinater_2d = pca.fit_transform(alle_vektorer)

sporsmal_2d = koordinater_2d[0]
db_2d = koordinater_2d[1:]

plt.figure(figsize=(8, 6))
# Plott databasetreff
plt.scatter(db_2d[:, 0], db_2d[:, 1], color='blue', label='Database-chunks', s=100)
for i, txt in enumerate([k['kilde'] for k in db_kilder]):
    plt.annotate(txt, (db_2d[i, 0], db_2d[i, 1]), fontsize=9, xytext=(5, 5), textcoords='offset points')

# Plott spørsmålet
plt.scatter(sporsmal_2d[0], sporsmal_2d[1], color='red', label='Spørsmål', s=150, marker='*')
plt.annotate("Spørsmål", (sporsmal_2d[0], sporsmal_2d[1]), fontsize=10, fontweight='bold', color='red', xytext=(5, 5), textcoords='offset points')

plt.title("Vektorrom-visualisering (PCA av RAG-embeddings)")
plt.xlabel("Dimensjon 1")
plt.ylabel("Dimensjon 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

filnavn = "vektorrom_visualisering.png"
plt.savefig(filnavn, bbox_inches='tight')
plt.close()

print(f"[Grafikk] Lagret bildefil: {filnavn}")