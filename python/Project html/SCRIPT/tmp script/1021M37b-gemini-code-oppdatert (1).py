import os
import sys
import warnings
import time
import re
from threading import Thread, Event

# --- Oppsett for norsk tekst i terminal ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging
import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# --- Konfigurasjon ---
SKRIPT_VERSJON = "v48-hybrid-sok-robust"
MAPPESTIER = [r"C:\working\python\dokumenter", r"C:\working\python\sensitiv"]
DB_STI = r"C:\working\python\chroma"
COLLECTION_NAVN = "mine_dokumenter"

# --- Initiering ---
hf_logging.set_verbosity_error()
chroma_client = chromadb.PersistentClient(path=DB_STI)
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Slett gammel samling for å starte "fresh" ved hver kjøring (valgfritt)
try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)

# --- Hjelpefunksjoner ---

def verifiser_database():
    """Sjekker om databasen faktisk har innhold."""
    antall = db_samling.count()
    print(f"\n[System] Databasestatus: {antall} chunks indeksert.")
    if antall > 0:
        eksempel = db_samling.peek(limit=1)
        print(f"[System] Eksempel på lagret data: {eksempel['documents'][0][:50]}...")
    else:
        print("[System] ADVARSEL: Databasen er tom! Sjekk mappestier.")

def _del_lang_tekst(tekst, maks_lengde, overlapp):
    setninger = re.split(r'(?<=[.!?])\s+', tekst.strip())
    biter = []
    naavaerende = ""
    for setning in setninger:
        if not setning: continue
        if len(naavaerende) + len(setning) + 1 <= maks_lengde:
            naavaerende = f"{naavaerende} {setning}".strip()
        else:
            if naavaerende: biter.append(naavaerende)
            overlapp_tekst = naavaerende[-overlapp:] if len(naavaerende) > overlapp else naavaerende
            naavaerende = f"{overlapp_tekst} {setning}".strip()
    if naavaerende: biter.append(naavaerende)
    return biter if biter else [tekst]

def del_tekst_i_chunks(tekst, maks_lengde=800, overlapp=150):
    tekst = tekst.strip()
    if not tekst: return [tekst]
    nummererte_deler = re.split(r'\n(?=\d+\.\s)', tekst)
    nummererte_deler = [d.strip() for d in nummererte_deler if d.strip()]
    if len(nummererte_deler) > 1:
        grunn_chunks = nummererte_deler
    else:
        avsnitt = re.split(r'\n\s*\n', tekst)
        avsnitt = [a.strip() for a in avsnitt if a.strip()]
        grunn_chunks = avsnitt if len(avsnitt) > 1 else [tekst]
    
    endelige_chunks = []
    for chunk in grunn_chunks:
        if len(chunk) <= maks_lengde: endelige_chunks.append(chunk)
        else: endelige_chunks.extend(_del_lang_tekst(chunk, maks_lengde, overlapp))
    return endelige_chunks if endelige_chunks else [tekst]

def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"): return
    filnavn = os.path.basename(full_sti)
    print(f"[System] Indekserer fil: {filnavn}")
    db_samling.delete(where={"kilde": filnavn})
    if not os.path.exists(full_sti): return
    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
        if not tekst: return
        chunks = del_tekst_i_chunks(tekst)
        dokumenter, metadatas, ids = [], [], []
        for indeks, chunk in enumerate(chunks):
            dokumenter.append(chunk)
            metadatas.append({"kilde": filnavn, "chunk_nr": indeks})
            ids.append(f"{filnavn}_chunk_{indeks}")
        db_samling.add(documents=dokumenter, metadatas=metadatas, ids=ids)
    except Exception as e:
        print(f"[Feil] Kunne ikke indeksere {filnavn}: {e}")

class DokumentLytter(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)
    def on_created(self, event):
        if not event.is_directory: oppdater_fil_i_database(event.src_path)

forste_indeksering_ferdig = Event()

def start_mappe_overvaking():
    observer = Observer()
    handler = DokumentLytter()
    for mappesti in MAPPESTIER:
        if not os.path.exists(mappesti): os.makedirs(mappesti)
        for rot, _, filer in os.walk(mappesti):
            for filnavn in filer: oppdater_fil_i_database(os.path.join(rot, filnavn))
        observer.schedule(handler, path=mappesti, recursive=True)
    observer.start()
    forste_indeksering_ferdig.set()

# --- AI-Modell ---
print("[System] Laster inn Qwen2.5-3B-Instruct...")
ai_modell = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", device=0 if torch.cuda.is_available() else -1, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
generasjons_konfig = GenerationConfig(max_new_tokens=400, do_sample=False, temperature=0.0)

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    # 1. Vektorsøk (Semantisk)
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=10)
    
    # 2. Hybrid-fallback (Keyword-søk)
    ord_i_sporsmal = [word.lower() for word in bruker_sporsmal.split() if len(word) > 3]
    alle_docs = db_samling.get()
    
    # Kombiner treff
    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    
    filtrerte_kilder = []
    # Legg til vektorsøk-treff
    for i in range(len(dokumenter)):
        utdrag = f"Fil: {metadataer[i]['kilde']} (Del {metadataer[i]['chunk_nr']})\nInnhold:\n{dokumenter[i]}"
        if utdrag not in filtrerte_kilder: filtrerte_kilder.append(utdrag)
    
    # Legg til keyword-treff
    for doc, meta in zip(alle_docs['documents'], alle_docs['metadatas']):
        if any(oi.lower() in doc.lower() for oi in ord_i_sporsmal):
            utdrag = f"Fil: {meta['kilde']} (Del {meta['chunk_nr']})\nInnhold:\n{doc}"
            if utdrag not in filtrerte_kilder: 
                print(f"[Hybrid-treff] Fant nøkkelord i {meta['kilde']}")
                filtrerte_kilder.append(utdrag)

    print(f"   - Sender {len(filtrerte_kilder)} unike tekstbiter til AI.")

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    meldinger = [
        {
            "role": "system",
            "content": (
                "Du er en konsis dokumentassistent. Svar utelukkende basert på de oppgitte kildene. "
                "Du skal alltid prioritere informasjonen i de oppgitte kildene. Hvis du er i tvil om informasjonen finnes, gå grundig gjennom alle kilde-utdragene en ekstra gang før du konkluderer."
                "Hvis du er i tvil om informasjonen finnes, gå grundig gjennom alle kilde-utdragene en ekstra gang. "
                "Svar direkte, korrekt og uten unødvendige høflighetsfraser."
                "Du er en konsis dokumentassistent. Du svarer utelukkende basert på de oppgitte kildene, "
                                "med naturlig, flytende norsk språk.\n\n"
                
                                "REGLER FOR SVAR:\n"
                                "1. Svar direkte på det som blir spurt om, uten å kommentere spørsmålsstillingen, "
                                "brukerens antakelser eller hvorfor de spør.\n"
                                "2. Hvis brukeren oppgir en verdi eller påstand som ikke stemmer med kildene, skal du "
                                "korrigere dette naturlig i svaret – bekreft at det er feil, og gi deretter den riktige "
                                "informasjonen i samme setning, i stedet for å bare svare ja eller nei.\n"
                                "3. Hvis kildene ikke inneholder svaret, si tydelig at informasjonen ikke finnes i kildene "
                                "du har tilgjengelig. Ikke gjett eller fyll inn med generell kunnskap.\n"
                                "4. Konkrete verdier som datoer, klokkeslett, tall og navn skal alltid gjengis EKSAKT slik "
                                "de står i kildeteksten. Ikke rund av, ikke generaliser, og ikke legg til årstall, "
                                "klokkeslett eller andre detaljer som ikke eksplisitt står i kilden.\n"
                                "5. Hvis du er usikker på om en verdi stemmer med kildene, skal du heller si at du er usikker "
                                "enn å gjette.\n"
                                "6. Ikke bruk faste innledningsfraser, maler eller unødvendige høflighetsfraser "
                                "(f.eks. 'Takk for spørsmålet', 'Basert på kildene kan jeg fortelle at...'). "
                                "Gå rett på svaret.\n"
                                "7. Hold svarene korte og presise – ikke lengre enn nødvendig for å svare fullstendig.\n"
                                "8. Hvis spørsmålet er tvetydig og kildene inneholder flere mulige svar, "
                                "spesifiser hvilket du svarer ut ifra, uten å gjette på hva brukeren mente."
            )
        },
        {
            "role": "user",
            "content": f"Spørsmål: {bruker_sporsmal}\n\nRelevante kilder:\n{kombinert_kontekst}"
        }
    ]

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    svar = resultat[0]["generated_text"].strip()
    
    print(f"-> Svar fra AI:\n{svar}")

if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()
    forste_indeksering_ferdig.wait(timeout=60)
    verifiser_database()

    print("\n[System] Klar til å ta imot spørsmål!")
    while True:
        try:
            sporsmal = input("Hva vil du spørre om? (eller 'avslutt'): ").strip()
            if sporsmal.lower() in ("avslutt", "exit", "quit"): break
            if not sporsmal: continue
            besvar_sporsmal(sporsmal)
        except KeyboardInterrupt: break