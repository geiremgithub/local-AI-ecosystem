import os
import sys
import warnings
import time
from threading import Thread, Event

# --- Oppsett for norsk tekst i terminal ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
# Forhindrer at tokenizere henger seg opp ved parallellkjøring
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v27-terminal-RAG-med-detaljert-grafikk"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

# Sjekk om GPU er tilgjengelig
if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16 # Bruk halv presisjon for raskere inferens på GPU
    print(f"[System] GPU funnet ({torch.cuda.get_device_name(0)}) - bruker CUDA.")
else:
    ENHET = "cpu"
    PRESISJON = torch.float32
    torch.set_num_threads(os.cpu_count())
    print(f"[System] Kjører på CPU med {os.cpu_count()} tråder.")

from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging
# Skjul unødvendige advarsler fra transformers
hf_logging.set_verbosity_error()

import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Biblioteker for visualisering av vektorer ---
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np # Nødvendig for vektor-håndtering

# --- Konfigurasjon ---
MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"

# --- Initialiser ChromaDB ---
# Bruker PersistentClient for å lagre databasen på disk
chroma_client = chromadb.PersistentClient(path=DB_STI)

# Velg en flerspråklig embedding-modell som støtter norsk
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAVN = "mine_dokumenter"

# Slett eksisterende samling for å starte "fresh" hver gang (valgfritt)
try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)

# --- Oppsett for tekst-splitting ---
tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,    # Størrelse på hver tekstbit (tegn)
    chunk_overlap=100, # Overlapp mellom biter for å bevare kontekst
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

# --- Funksjoner ---

def del_tekst_i_chunks(tekst):
    """Deler opp en lang tekststreng i mindre biter."""
    return tekst_splitter.split_text(tekst)

def oppdater_fil_i_database(full_sti):
    """Indekserer en enkelt .txt fil i ChromaDB."""
    if not full_sti.endswith(".txt"):
        return
    filnavn = os.path.basename(full_sti)
    # Fjern gamle versjoner av dokumentet hvis det finnes
    try:
        db_samling.delete(where={"kilde": filnavn})
    except Exception:
        pass

    if not os.path.exists(full_sti):
        return

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
    except Exception:
        return

    if not tekst:
        return

    # Del opp og legg til i databasen
    chunks = del_tekst_i_chunks(tekst)
    dokumenter, metadatas, ids = [], [], []

    for indeks, chunk in enumerate(chunks):
        dokumenter.append(chunk)
        metadatas.append({"kilde": filnavn, "chunk_nr": indeks})
        ids.append(f"{filnavn}_chunk_{indeks}")

    db_samling.add(documents=dokumenter, metadatas=metadatas, ids=ids)
    print(f"[Database] Indeksert {filnavn} ({len(chunks)} chunks).")

# --- Watchdog Event Handler ---
class DokumentLytter(FileSystemEventHandler):
    """Lytter etter endringer i mappen og oppdaterer databasen automatisk."""
    def on_modified(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)
    def on_created(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)

forste_indeksering_ferdig = Event()

def start_mappe_overvaking():
    """Initialiserer overvåking av dokument-mappen."""
    # Opprett mappen hvis den ikke finnes
    if not os.path.exists(MAPPESTI): os.makedirs(MAPPESTI)
    # Indekser eksisterende filer først
    print(f"[System] Indekserer filer i {MAPPESTI}...")
    for filnavn in os.listdir(MAPPESTI):
        oppdater_fil_i_database(os.path.join(MAPPESTI, filnavn))
    print(f"[System] Første indeksering ferdig.")

    # Start lytteren
    handler = DokumentLytter()
    observer = Observer()
    observer.schedule(handler, path=MAPPESTI, recursive=False)
    observer.start()
    forste_indeksering_ferdig.set()
    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- Last inn AI Modell (LLM) ---
print("[System] Laster inn Qwen2.5-3B-Instruct...")
try:
    ai_modell = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device=0 if ENHET == "cuda" else -1, # Bruk GPU hvis tilgjengelig
        torch_dtype=PRESISJON,
    )
    print("[System] AI-modell lastet.")
except Exception as e:
    print(f"[Feil] Kunne ikke laste AI-modell: {e}")
    sys.exit(1)

generasjons_konfig = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,      # Tillat variasjon i svaret
    temperature=0.3,     # Lav temperatur for presise svar
)

def generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander):
    """
    Genererer en forbedret 2D PCA-visualisering av vektorrommet
    med tekstetiketter for hver chunk og spørsmål.
    """
    print("[Grafikk] Genererer visualisering...")
    
    # 1. Hent vektorer for spørsmål og dokumenter
    try:
        sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
        dokument_vektorer = embedding_modell(dokumenter)
    except Exception as e:
        print(f"[Feil] Kunne ikke generere vektorer for grafikk: {e}")
        return

    # Kombiner alle vektorer for PCA
    alle_vektorer = [sporsmal_vektor] + list(dokument_vektorer)
    alle_vektorer = np.array(alle_vektorer)

    # 2. Reduser dimensjonaliteten til 2D med PCA
    pca = PCA(n_components=2)
    try:
        reduserte_vektorer = pca.fit_transform(alle_vektorer)
    except ValueError:
        print("[Feil] For få punkter til å kjøre PCA.")
        return
    
    sporsmal_2d = reduserte_vektorer[0]
    dokumenter_2d = reduserte_vektorer[1:]
    
    # 3. Opprett plottet
    plt.figure(figsize=(14, 10)) # Større figur for bedre lesbarhet
    
    # Definer farger for kilder (hvis du har flere filer)
    unike_kilder = list(set([m.get("kilde") for m in metadataer]))
    farger = plt.cm.viridis(np.linspace(0, 1, len(unike_kilder)))
    kilde_farge_map = {kilde: farger[i] for i, kilde in enumerate(unike_kilder)}

    # 4. Tegn dokument-chunkene
    for i, dok_2d in enumerate(dokumenter_2d):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        # Henter ut de første 40 tegnene av teksten som en smakebit
        tekst_smakebit = dokumenter[i][:40].replace("\n", " ")
        dist = avstander[i]
        
        farge = kilde_farge_map.get(kilde, 'blue')
        
        # Plott punktet
        plt.scatter(dok_2d[0], dok_2d[1], color=farge, s=120, alpha=0.8, label=f{kilde})
        
        # Lag en informativ etikett med tekstinnhold
        etikett = f"[{kilde} - C{chunk_nr}]\n\"{tekst_smakebit}...\"\n(Avst: {dist:.4f})"
        
        # Plasser teksten med en fin boks rundt
        plt.text(dok_2d[0] + 0.02, dok_2d[1], etikett, fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor=farge, boxstyle='round,pad=0.3'))
        
    # 5. Tegn spørsmålet ditt som en rød stjerne
    plt.scatter(sporsmal_2d[0], sporsmal_2d[1], color='red', s=300, marker='*', zorder=5, label='Ditt Spørsmål')
    
    # Etikett for spørsmålet
    sporsmal_etikett = f"Spørsmål:\n\"{bruker_sporsmal}\""
    plt.text(sporsmal_2d[0] + 0.02, sporsmal_2d[1], sporsmal_etikett, fontsize=10, fontweight='bold', color='red',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'), zorder=6)
    
    # 6. Pynt opp plottet
    plt.title(f"Vektorrom-visualisering av RAG-søk\n(Topp 5 treff)", fontsize=14, fontweight='bold')
    plt.xlabel("PCA Dimensjon 1", fontsize=12)
    plt.ylabel("PCA Dimensjon 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Legg til en enkel forklaring (legende) for kildene
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), title="Kildedokumenter")
    
    # 7. Lagre figuren
    filnavn = "vektorrom_visualisering.png"
    try:
        plt.savefig(filnavn, dpi=300, bbox_inches='tight') # bbox_inches='tight' får med all tekst
        plt.close()
        print(f"[Grafikk] Lagret forbedret visualisering som bildefil: {filnavn}")
    except Exception as e:
        print(f"[Feil] Kunne ikke lagre bildefil: {e}")


def besvar_sporsmal(bruker_sporsmal):
    """
    Hovedfunksjon for RAG:
    1. Søk i database
    2. Generer grafikk
    3. Generer AI-svar
    """
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    # 1. Tokenisering med fiks for norske tegn (debug-utskrift)
    sporsmal_tokens = ai_modell.tokenizer.tokenize(bruker_sporsmal)
    fiksede_tokens = [t.encode('latin1', errors='ignore').decode('utf-8', errors='ignore') for t in sporsmal_tokens]
    
    print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(sporsmal_tokens)} tokens.")
    print(f"   - Tokens (ord/stavelser): {fiksede_tokens}")

    # 2. Embedding (vektorisering) av spørsmål
    print(f"2. Embedding: Spørsmålet ble omgjort til en vektor.")

    # 3. Databasesøk
    print("3. Databasesøk: Finner relevante dokument-chunks...")
    try:
        sok_resultat = db_samling.query(
            query_texts=[bruker_sporsmal],
            n_results=5 # Hent topp 5 treff for visualisering
        )
    except Exception as e:
        print(f"[Feil] Databasesøk feilet: {e}")
        return
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0] if "distances" in sok_resultat else [0.0] * len(dokumenter)

    # Debug-utskrift av treff
    print("   - Topp treff fra databasen:")
    for i in range(len(dokumenter)):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        avst = avstander[i]
        tekst_smakebit = dokumenter[i][:30].replace("\n", " ") + "..."
        print(f"     [{i}] Kilde: {kilde} (Chunk {chunk_nr}) | Avstand: {avst:.4f} | Tekst: {tekst_smakebit}")

    # --- NYTT: Kall funksjonen for å generere den detaljerte bildefilen ---
    generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander)

    # 4. LLM-Generering: Send kontekst og spørsmål til AI-modellen
    print("4. LLM-Generering: Sender prompt med kontekst til Qwen-modellen...")
    
    # Bruk kun topp 3 treff som kontekst for selve svaret
    kombinert_kontekst = "\n\n".join(dokumenter[:3])

    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er