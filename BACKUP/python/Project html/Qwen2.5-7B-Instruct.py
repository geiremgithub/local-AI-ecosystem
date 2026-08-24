import os
import re
import sys
import warnings
import time
from threading import Thread, Event

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v18-terminal-Qwen2.5-7B"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16
    print(f"[System] GPU funnet ({torch.cuda.get_device_name(0)}) - bruker CUDA.")
else:
    ENHET = "cpu"
    PRESISJON = torch.float32 # 7B på CPU bruker mye minne, hold deg til float32 eller bfloat16
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

print("[System] Laster inn Qwen2.5-7B-Instruct (dette kan ta litt tid)...")
# Oppgradert modell-linje:
ai_modell = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct", 
    device_map="auto" if ENHET == "cuda" else None,
    device=None if ENHET == "cuda" else 0,
    torch_dtype=torch.bfloat16 if ENHET == "cuda" else torch.float32,
)

generasjons_konfig = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)

SAMTALE_HISTORIKK = []
MAKS_HISTORIKK = 2

def besvar_sporsmal(bruker_sporsmal):
    print("Søker i dokumenter...")
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    dokument_kontekster = sok_resultat["documents"][0]
    
    kombinert_kontekst = "\n\n".join(dokument_kontekster)
    
    meldinger = [
        {"role": "system", "content": f"Du er en hjelpsom assistent. Bruk følgende info:\n{kombinert_kontekst}"},
    ]
    meldinger.append({"role": "user", "content": bruker_sporsmal})

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)

    svar = resultat[0]["generated_text"].strip()
    print(f"-> Svar: {svar}\n")

if __name__ == "__main__":
    start_mappe_overvaking()
    print("\n[System] Klar!")
    while True:
        try:
            sporsmal = input("Spørsmål: ")
            if sporsmal.lower() in ("avslutt", "exit"): break
            besvar_sporsmal(sporsmal)
        except KeyboardInterrupt: break