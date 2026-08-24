import os
import warnings
import time
from threading import Thread

# Skru av unødvendige advarsler i terminalen
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# Stier til mapper
# ---------------------------------------------------------
MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"

# ---------------------------------------------------------
# 1. Oppsett av Chroma-database
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# LangChain Recursive Chunking
# ---------------------------------------------------------
def del_tekst_i_chunks_rekursiv(
    tekst,
    chunk_storrelse=1000,
    overlapping=200
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_storrelse,
        chunk_overlap=overlapping,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(tekst)

# ---------------------------------------------------------
# Oppdaterer én fil i databasen
# ---------------------------------------------------------
def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith('.txt'):
        return

    filnavn = os.path.basename(full_sti)

    try:
        db_samling.delete(where={'kilde': filnavn})
    except Exception:
        pass

    if not os.path.exists(full_sti):
        print(f'[Database] Fjernet {filnavn} (filen ble slettet).')
        return

    with open(full_sti, 'r', encoding='utf-8') as fil:
        tekst = fil.read().strip()

    if not tekst:
        return

    chunks = del_tekst_i_chunks_rekursiv(
        tekst,
        chunk_storrelse=1000,
        overlapping=200
    )

    print(f'[Chunking] Opprettet {len(chunks)} chunks')

    dokumenter = []
    metadatas = []
    ids = []

    for indeks, chunk in enumerate(chunks):
        dokumenter.append(chunk)
        metadatas.append({'kilde': filnavn, 'chunk_nr': indeks})
        ids.append(f'{filnavn}_chunk_{indeks}')

    db_samling.add(
        documents=dokumenter,
        metadatas=metadatas,
        ids=ids
    )

    print(f'[Database] Suksess! Delte {filnavn} inn i {len(chunks)} biter med LangChain chunking.')

class DokumentLytter(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)


def start_mappe_overvaking():
    print('[System] Gjør første skanning og oppretter biter...')

    if not os.path.exists(MAPPESTI):
        os.makedirs(MAPPESTI)

    for filnavn in os.listdir(MAPPESTI):
        oppdater_fil_i_database(os.path.join(MAPPESTI, filnavn))

    handler = DokumentLytter()
    observer = Observer()
    observer.schedule(handler, path=MAPPESTI, recursive=False)
    observer.start()

    print(f'[Watchdog] Lytter aktivt i: {MAPPESTI}')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
bakgrunns_lytter.start()

time.sleep(2)

print('[System] Laster inn Qwen-modellen...')
ai_modell = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct')

print('RAG-system startet med LangChain RecursiveCharacterTextSplitter.')
