import os
import warnings
import time
from threading import Thread

# Skrur av unødvendige advarsler i terminalen
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Stier til mapper
MAPPESTIE = r"C:\working\python\dokumenter"
DB_STIE = r"C:\working\python\chroma_db"

# 1. Sett opp lokal Vektordatabase (ChromaDB) med en NORSK/FLERSPRÅKLIG embedding-modell
chroma_client = chromadb.PersistentClient(path=DB_STIE)

# FIKS: Vi bruker en modell som forstår norsk semantikk og språk perfekt!
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Sletter gammel samling for å tvinge frem en ren re-indeksering med den nye modellen
try:
    chroma_client.delete_collection(name="mine_dokumenter")
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name="mine_dokumenter", 
    embedding_function=embedding_modell
)

# Deler tekst inn i chunks med overlapping basert på ord
def del_tekst_i_chunks(tekst, chunk_storrelse=100, overlapping=20):
    ord_liste = tekst.split()
    chunks = []
    
    i = 0
    while i < len(ord_liste):
        chunk_ord = ord_liste[i : i + chunk_storrelse]
        chunk_tekst = " ".join(chunk_ord)
        chunks.append(chunk_tekst)
        
        if i + chunk_storrelse >= len(ord_liste):
            break
            
        i += (chunk_storrelse - overlapping)
        
    return chunks

# Oppdatert funksjon for å håndtere CHUNKS i databasen
def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"):
        return
    filnavn = os.path.basename(full_sti)
    
    try:
        db_samling.delete(where={"kilde": filnavn})
    except Exception:
        pass

    if not os.path.exists(full_sti):
        print(f"[Database] Slettet {filnavn} fra databasen.")
        return

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
            
        if tekst:
            chunks = del_tekst_i_chunks(tekst, chunk_storrelse=80, overlapping=20)
            
            dokumenter = []
            metadatas = []
            ids = []
            
            for index, chunk in enumerate(chunks):
                dokumenter.append(chunk)
                metadatas.append({"kilde": filnavn, "chunk_nr": index})
                ids.append(f"{filnavn}_chunk_{index}")
            
            db_samling.upsert(
                documents=dokumenter,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[Database] Suksess! Delt {filnavn} inn i {len(chunks)} chunks med overlapping.")
    except Exception as e:
        print(f"[Database] Feil ved lesing av {filnavn}: {e}")

# 2. Sett opp Watchdog (Mappe-overvåker)
class DokumentLytter(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)
    def on_created(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)
    def on_deleted(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)

def start_mappe_overvaking():
    print("[System] Gjør første skanning og oppretter chunks...")
    if not os.path.exists(MAPPESTIE):
        os.makedirs(MAPPESTIE)
    for filnavn in os.listdir(MAPPESTIE):
        oppdater_fil_i_database(os.path.join(MAPPESTIE, filnavn))
        
    handler = DokumentLytter()
    observer = Observer()
    observer.schedule(handler, path=MAPPESTIE, recursive=False)
    observer.start()
    print(f"[Watchdog] Lytter aktivt i: {MAPPESTIE}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
bakgrunns_lytter.start()

# 3. Last inn språkmodellen Qwen (Generatoren)
print("[System] Laster inn Qwen-modellen...")
ai_modell = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")

# 4. Spørsmåls-løkke
while True:
    time.sleep(0.5)
    bruker_sporsmal = input("\nHva vil du spørre om? (eller skriv 'avslutt'): ")
    
    if bruker_sporsmal.lower() == 'avslutt':
        print("Avslutter programmet. Ha en fin dag!")
        break
        
    if not bruker_sporsmal.strip():
        continue
        
    print("Søker etter mest relevante tekst-chunk...")
    sok_resultat = db_samling.query(
        query_texts=[bruker_sporsmal],
        n_results=1
    )
    
    if sok_resultat['documents'] and sok_resultat['documents'][0]:
        dokument_kontekst = sok_resultat['documents'][0][0]
        metadata = sok_resultat['metadatas'][0][0]
        kilde_fil = metadata['kilde']
        chunk_nr = metadata['chunk_nr']
    else:
        print("-> Svar fra AI: Fant ingen relevante dokumenter.")
        continue
        
    print(f"AI-en analyserer chunk nr {chunk_nr} fra filen: {kilde_fil}...")
    
    meldinger = [
        {"role": "system", "content": f"Du er en presis assistent. Svar kort med én setning på norsk basert KUN på denne teksten:\n{dokument_kontekst}\nHvis du ikke finner svaret, svarer du nøyaktig: 'Jeg finner ikke svar i dokumentet.'"},
        {"role": "user", "content": bruker_sporsmal}
    ]
    
    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)
    
    resultat = ai_modell(
        prompt, 
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False, 
        clean_up_tokenization_spaces=False
    )
    
    full_generert_tekst = resultat[0]['generated_text']
    svar_linjer = full_generert_tekst.strip().split("\n")
    svar = svar_linjer[0].strip() if svar_linjer else ""
    
    if svar and not svar.endswith("."):
        svar += "."

    print("\n")
    print(f"-> Svar fra AI: [{kilde_fil} | Chunk {chunk_nr}] {svar}")
    print("\n")
