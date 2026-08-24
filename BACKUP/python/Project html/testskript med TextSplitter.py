import os
import sys
import warnings
import time
from threading import Thread, Event

# --- Oppsett for norsk tekst i terminal ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v61-stabil-med-unik-kilde-sortering"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16
    print(f"[System] GPU funnet ({torch.cuda.get_device_name(0)}) - bruker CUDA.")
else:
    ENHET = "cpu"
    PRESISJON = torch.float32
    torch.set_num_threads(os.cpu_count())
    print(f"[System] Kjører på CPU med {os.cpu_count()} tråder.")

from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Konfigurasjon av mapper som skal overvåkes ---
MAPPESTIER = [
    r"C:\working\python\dokumenter",
    r"C:\working\python\sensitiv"
]
DB_STI = r"C:\working\python\chroma"

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

# Beholder tekst-splittingen
tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

def del_tekst_i_chunks(tekst):
    return tekst_splitter.split_text(tekst)

def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"):
        return
    filnavn = os.path.basename(full_sti)
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

    chunks = del_tekst_i_chunks(tekst)
    dokumenter, metadatas, ids = [], [], []

    for indeks, chunk in enumerate(chunks):
        dokumenter.append(chunk)
        metadatas.append({"kilde": filnavn, "chunk_nr": indeks})
        ids.append(f"{filnavn}_chunk_{indeks}")

    db_samling.add(documents=dokumenter, metadatas=metadatas, ids=ids)
    print(f"[Database] Indeksert {filnavn} ({len(chunks)} chunks).")

class DokumentLytter(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)
    def on_created(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)

forste_indeksering_ferdig = Event()

def start_mappe_overvaking():
    observer = Observer()
    handler = DokumentLytter()

    for mappe in MAPPESTIER:
        if not os.path.exists(mappe):
            os.makedirs(mappe)

        for rot, _, filer in os.walk(mappe):
            for filnavn in filer:
                full_sti = os.path.join(rot, filnavn)
                oppdater_fil_i_database(full_sti)

        observer.schedule(handler, path=mappe, recursive=True)
        print(f"[System] Overvåker mappe (inkl. undermapper): {mappe}")

    observer.start()
    forste_indeksering_ferdig.set()

print("[System] Laster inn Qwen2.5-3B-Instruct...")
ai_modell = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct", 
    device=0 if ENHET == "cuda" else -1,
    torch_dtype=PRESISJON,
)

generasjons_konfig = GenerationConfig(
    max_new_tokens=400,
    do_sample=False,
    temperature=0.0,
)

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    tokens = ai_modell.tokenizer.tokenize(bruker_sporsmal)
    token_ids = ai_modell.tokenizer.encode(bruker_sporsmal, add_special_tokens=False)
    lesbare_tokens = [t.replace('Ġ', ' ').replace('Ã¥', 'å') for t in tokens]
    
    print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(tokens)} tokens.")
    print(f"   - Tokens: {lesbare_tokens}")
    print(f"   - Token ID-er: {token_ids}")

    sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
    forste_10_vektorer = [round(v, 4) for v in sporsmal_vektor[:10]]
    print(f"2. Embedding (Spørsmål-vektor): Totalt {len(sporsmal_vektor)} dimensjoner.")
    print(f"   - Vektortall (første 10): {forste_10_vektorer} ...")

    print("3. Databasesøk: Sorterer chunks basert på matchet avstand i vektordataen...")
    sok_resultat = db_samling.query(
        query_texts=[bruker_sporsmal], 
        n_results=15, 
        include=["documents", "metadatas", "distances", "embeddings"]
    )
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0]
    vektorer = sok_resultat["embeddings"][0]

    print("   - Topp treff fra databasen og deres vektorer:")
    filtrerte_kilder = []
    sett_filer = set()
    
    for i in range(len(dokumenter)):
        kilde = metadataer[i].get('kilde', 'ukjent')
        chunk_nr = metadataer[i].get('chunk_nr', 0)
        avstand = avstander[i]
        forste_8_vektor = [round(v, 4) for v in vektorer[i][:8]]
        
        print(f"      [{i}] Kilde: {kilde} (Chunk {chunk_nr}) | Avstand: {avstand:.4f}")
        print(f"          Vektortall (første 8): {forste_8_vektor} ...")
        
        if kilde not in sett_filer or len(sett_filer) < 3:
            filtrerte_kilder.append(f"Fil: {kilde}\nInnhold:\n{dokumenter[i]}")
            sett_filer.add(kilde)

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en presis norsk assistent for dokumentanalyse. "
                "Analyser samtlige oppgitte kilder nedenfor nøye. "
                "Sjekk hver enkelt fil for om den omtaler temaet eller søkeordet det spørres om. "
                "List opp alle filer som inneholder relevant informasjon, trekk ut sitatene, "
                "og unngå å påstå at en fil mangler informasjon hvis den faktisk finnes i kildene."
            )
        },
        {
            "role": "user", 
            "content": f"Spørsmål: {bruker_sporsmal}\n\nRelevante kilder:\n{kombinert_kontekst}"
        }
    ]

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)

    print("\n4. Sender renset prompt til Qwen-modellen...")
    start_tid = time.time()
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    slutt_tid = time.time()
    
    total_tid = slutt_tid - start_tid
    svar = resultat[0]["generated_text"].strip()
    
    genererte_tokens = len(ai_modell.tokenizer.encode(svar, add_special_tokens=False))
    tokens_per_sekund = genererte_tokens / total_tid if total_tid > 0 else 0

    print(f"\n-> Svar fra AI:\n{svar}")
    print(f"[Ytelse] Generering tok {total_tid:.2f} sekunder (ca. {tokens_per_sekund:.1f} tokens/sek)\n")

if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    forste_indeksering_ferdig.wait(timeout=60)

    print("\n[System] Klar til å ta imot spørsmål i terminalen!")
    while True:
        try:
            sporsmal = input("Hva vil du spørre om? (eller skriv 'avslutt'): ").strip()
            if sporsmal.lower() in ("avslutt", "exit", "quit"): 
                break
            if not sporsmal:
                continue
            besvar_sporsmal(sporsmal)
        except KeyboardInterrupt: 
            break