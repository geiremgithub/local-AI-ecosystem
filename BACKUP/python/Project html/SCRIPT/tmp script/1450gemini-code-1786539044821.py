import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def behandle_sporsmal(sporsmalstekst: str):
    print(f"=== [HENDELSE: NYTT SPØRSMÅL STILT] ===")
    print(f"Spørsmål mottatt: \"{sporsmalstekst}\"\n")
    
    # 1. Tokenisering skjer nå, som direkte konsekvens av at spørsmålet ble stilt
    print("1. [TOKENISERING]")
    tokens = sporsmalstekst.split()
    token_ids = [hash(t) % 10000 for t in tokens]
    print(f"   - Genererte {len(tokens)} tokens fra det stilte spørsmålet.")
    print(f"   - Tokens: {tokens}")
    print(f"   - Token ID-er: {token_ids}\n")
    
    # 2. Embedding av det stilte spørsmålet
    print("2. [EMBEDDING]")
    modell = SentenceTransformer('all-MiniLM-L6-v2')
    sporsmal_vektor = modell.encode(sporsmalstekst)
    print(f"   - Spørsmålet ble vektorisert til {len(sporsmal_vektor)} dimensjoner.")
    print(f"   - Vektortall (utdrag): {[round(x, 4) for x in sporsmal_vektor[:6]]} ...\n")
    
    # 3. Databasetøk og henting av databasens vektorer basert på det stilte spørsmålet
    print("3. [DATABASESØK OG VEKTORER]")
    lokal_database = [
        {"kilde": "Reglement_kap1.txt", "tekst": "Retningslinjer for intern dataflyt og sikkerhet."},
        {"kilde": "Sikkerhetsmanual.txt", "tekst": "Krav til kryptering og null-lagring (zero data retention)."},
        {"kilde": "Arkitektur_hybrid.txt", "tekst": "Beskrivelse av hybrid sky og lokal datalagring."}
    ]
    
    db_tekster = [item["tekst"] for item in lokal_database]
    db_vektorer = modell.encode(db_tekster)
    
    avstander = [np.linalg.norm(sporsmal_vektor - v) for v in db_vektorer]
    sortert_indeks = np.argsort(avstander)
    
    print("   Søker i databasen etter treff knyttet til det stilte spørsmålet:")
    for i, idx in enumerate(sortert_indeks):
        treff = lokal_database[idx]
        dist = avstander[idx]
        vektor_utdrag = [round(x, 4) for x in db_vektorer[idx][:6]]
        print(f"   [{i+1}] Kilde: {treff['kilde']} (Avstand: {dist:.4f})")
        print(f"       Vektortall: {vektor_utdrag} ...")
    print()
    
    # 4. Vektorrom-visualisering basert på det stilte spørsmålet
    print("4. [VEKTORROM-VISUALISERING]")
    alle_vektorer = np.vstack([sporsmal_vektor, db_vektorer])
    pca = PCA(n_components=2)
    koordinater = pca.fit_transform(alle_vektorer)
    
    sporsmal_2d = koordinater[0]
    db_2d = koordinater[1:]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(db_2d[:, 0], db_2d[:, 1], color='blue', label='Databasetreff', s=100)
    for i, item in enumerate(lokal_database):
        plt.annotate(item['kilde'], (db_2d[i, 0], db_2d[i, 1]), fontsize=9, xytext=(5, 5), textcoords='offset points')
        
    plt.scatter(sporsmal_2d[0], sporsmal_2d[1], color='red', label='Stilt Spørsmål', s=150, marker='*')
    plt.annotate("Spørsmål", (sporsmal_2d[0], sporsmal_2d[1]), fontsize=10, fontweight='bold', color='red', xytext=(5, 5), textcoords='offset points')
    
    plt.title(f"Vektorrom for spørsmål: \"{sporsmalstekst}\"")
    plt.xlabel("Dimensjon 1")
    plt.ylabel("Dimensjon 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filnavn = "vektorrom_etter_sporsmal.png"
    plt.savefig(filnavn, bbox_inches='tight')
    plt.close()
    print(f"   - Vektorrom-visualisering generert og lagret som: {filnavn}")

# Kjører prosessen utelukkende ETTER at et spørsmål er stilt:
if __name__ == "__main__":
    stilte_sporsmal = "Hvordan fungerer sikkerhet og lagring i en hybrid løsning?"
    behandle_sporsmal(stilte_sporsmal)