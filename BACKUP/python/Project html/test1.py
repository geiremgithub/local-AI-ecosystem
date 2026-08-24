import os
import sys
import warnings
import time
from threading import Thread, Event
import torch
from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging
import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# --- Oppsett ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()

MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"

# --- Initialisering ---
chroma_client = chromadb.PersistentClient(path=DB_STI)
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION_NAVN = "mine_dokumenter"

try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN, 
    embedding_function=embedding_modell
)

tekst_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100)

# --- Funksjon for grafikk ---
def generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander):
    print("[Grafikk] Genererer detaljert visualisering...")
    sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
    dokument_vektorer = embedding_modell(dokumenter)
    alle_vektorer = np.array([sporsmal_vektor] + list(dokument_vektorer))
    
    pca = PCA(n_components=2)
    reduserte = pca.fit_transform(alle_vektorer)
    
    plt.figure(figsize=(14, 10))
    # Tegn chunks
    for i in range(len(dokumenter)):
        k = metadataer[i].get("kilde", "ukjent")
        c = metadataer[i].get("chunk_nr", 0)
        smakebit = dokumenter[i][:40].replace("\n", " ") + "..."
        etikett = f"[{k}-C{c}]\n{smakebit}\n(Avst: {avstander[i]:.4f})"
        plt.scatter(reduserte[i+1][0], reduserte[i+1][1], s=120, alpha=0.7)
        plt.text(reduserte[i+1][0]+0.02, reduserte[i+1][1], etikett, fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Tegn spørsmål
    plt.scatter(reduserte[0][0], reduserte[0][1], color='red', s=300, marker='*', label='Spørsmål')
    plt.title("Vektorrom-visualisering av RAG-søk", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.savefig("vektorrom_visualisering.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Grafikk] Lagret vektorrom_visualisering.png")

# --- Hovedfunksjoner ---
def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"): return
    filnavn = os.path.basename(full_sti)
    try:
        db_samling.delete(where={"kilde": filnavn})
        with open(full_sti, "r", encoding="utf-8") as f:
            tekst = f.read().strip()
        chunks = tekst_splitter.split_text(tekst)
        db_samling.add(
            documents=chunks,
            metadatas=[{"kilde": filnavn, "chunk_nr": i} for i in range(len(chunks))],
            ids=[f"{filnavn}_chunk_{i}" for i in range(len(chunks))]
        )
        print(f"[Database] Indeksert {filnavn}")
    except Exception: pass

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    
    if not sok_resultat["documents"][0]:
        print("-> Fant ingen dokumenter.")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0]
    
    # Generer grafikk med detaljer
    generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander)
    
    print("-> Svar: Sjekk fila 'vektorrom_visualisering.png' for detaljert oversikt!")

if __name__ == "__main__":
    # Indekser mappe
    if os.path.exists(MAPPESTI):
        for f in os.listdir(MAPPESTI):
            oppdater_fil_i_database(os.path.join(MAPPESTI, f))
            
    print("\n[System] Klar til å ta imot spørsmål!")
    while True:
        try:
            sporsmal = input("\nHva vil du spørre om? (avslutt for å stoppe): ")
            if sporsmal.lower() == "avslutt": break
            besvar_sporsmal(sporsmal)
        except KeyboardInterrupt: break