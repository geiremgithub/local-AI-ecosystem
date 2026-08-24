import os
import sys
import warnings
import time
from threading import Thread, Event

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v25-terminal-RAG-norske-tegn"
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

MAPPESTI = r"C:\working\python\dokumenter"
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

tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=100,
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
    if not os.path.exists(MAPPESTI): os.makedirs(MAPPESTI)
    for filnavn in os.listdir(MAPPESTI):
        oppdater_fil_i_database(os.path.join(MAPPESTI, filnavn))

    handler = DokumentLytter()
    observer = Observer()
    observer.schedule(handler, path=MAPPESTI, recursive=False)
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
    max_new_tokens=200,
    do_sample=True,
    temperature=0.3,
)

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    # 1. Tokenisering med fiks for norske tegn
    sporsmal_tokens = ai_modell.tokenizer.tokenize(bruker_sporsmal)
    fiksede_tokens = [t.encode('latin1', errors='ignore').decode('utf-8', errors='ignore') for t in sporsmal_tokens]
    sporsmal_token_ids = ai_modell.tokenizer.encode(bruker_sporsmal)
    
    print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(sporsmal_tokens)} tokens.")
    print(f"   - Tokens (ord/stavelser): {fiksede_tokens}")
    print(f"   - Token ID-er: {sporsmal_token_ids}")

    # 2. Embedding (vektorisering)
    vektorer = embedding_modell([bruker_sporsmal])
    forste_vektor_tall = vektorer[0][:5]
    print(f"2. Embedding: Spørsmålet ble omgjort til en vektor med {len(vektorer[0])} dimensjoner.")
    print(f"   - Eksempel på vektor-tall (første 5 av {len(vektorer[0])}): {[round(t, 4) for t in forste_vektor_tall]} ...")

    print("3. Databasesøk: Sorterer chunks basert på matchet avstand i vektordataen...")
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0] if "distances" in sok_resultat else [0.0] * len(dokumenter)

    print("   - Topp treff fra databasen:")
    for i in range(min(3, len(dokumenter))):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        avst = avstander[i]
        print(f"     [{i}] Kilde: {kilde} (Chunk {chunk_nr}) | Avstand: {avst:.4f}")

    kombinert_kontekst = "\n\n".join(dokumenter[:3])

    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en presis assistent. Svar KUN direkte på det brukeren spør om basert på teksten under. "
                "Ikke ta med ekstra uoppfordret informasjon.\n\n"
                f"----- DOKUMENTINNHOLD -----\n{kombinert_kontekst}"
            )
        },
    ]
    meldinger.append({"role": "user", "content": bruker_sporsmal})

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)

    print("4. LLM-Generering: Sender prompt med kontekst til Qwen-modellen...")
    start_tid = time.time()
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    slutt_tid = time.time()
    
    total_tid = slutt_tid - start_tid
    svar = resultat[0]["generated_text"].strip()
    
    antall_ord = len(svar.split())
    antall_tokens_omtrent = int(antall_ord * 1.3)
    hastighet = antall_tokens_omtrent / total_tid if total_tid > 0 else 0

    beste_kilde = metadataer[0].get("kilde", "ukjent")
    beste_chunk = metadataer[0].get("chunk_nr", 0)
    
    print(f"-> Svar fra AI: [{beste_kilde} | Chunk {beste_chunk}] {svar}")
    print(f"[Ytelse] Generering tok {total_tid:.2f} sekunder (ca. {hastighet:.1f} tokens/sek)\n")

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