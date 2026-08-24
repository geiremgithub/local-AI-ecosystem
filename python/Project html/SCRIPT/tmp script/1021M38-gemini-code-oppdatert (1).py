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

SKRIPT_VERSJON = "v47-renset-system-prompt"
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

MAPPESTIER = [
    r"C:\working\python\dokumenter",
    r"C:\working\python\sensitiv"
]
DB_STI = r"C:\working\python\chroma"
COLLECTION_NAVN = "mine_dokumenter"

chroma_client = chromadb.PersistentClient(path=DB_STI)
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)

# --- Determinisk søk (Ingen AI-hallusinasjon ved oversiktsspørsmål) ---
def sok_etter_ord_i_filer(nøkkelord):
    print(f"\n--- [SØKER ETTER '{nøkkelord}' DIREKTE I DATABASEN] ---")
    alle_data = db_samling.get()
    funn = 0
    
    for doc, meta in zip(alle_data['documents'], alle_data['metadatas']):
        if nøkkelord.lower() in doc.lower():
            funn += 1
            print(f"- **{meta['kilde']} (Del {meta['chunk_nr']})**: funnet!")
            utdrag = doc.strip().replace("\n", " ")
            print(f"  Innhold: \"{utdrag[:100]}...\"")
            
    if funn == 0:
        print(f"Fant ingen forekomster av '{nøkkelord}'.")
    else:
        print(f"\nTotalt {funn} treff funnet.")

# --- Hjelpefunksjoner for chunking og indeksering ---
def _del_lang_tekst(tekst, maks_lengde, overlapp):
    setninger = re.split(r'(?<=[.!?])\s+', tekst.strip())
    biter = []
    naavaerende = ""

    for setning in setninger:
        if not setning:
            continue
        if len(naavaerende) + len(setning) + 1 <= maks_lengde:
            naavaerende = f"{naavaerende} {setning}".strip()
        else:
            if naavaerende:
                biter.append(naavaerende)
            overlapp_tekst = naavaerende[-overlapp:] if len(naavaerende) > overlapp else naavaerende
            naavaerende = f"{overlapp_tekst} {setning}".strip()

    if naavaerende:
        biter.append(naavaerende)

    return biter if biter else [tekst]

def del_tekst_i_chunks(tekst, maks_lengde=800, overlapp=150):
    tekst = tekst.strip()
    if not tekst:
        return [tekst]

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
        if len(chunk) <= maks_lengde:
            endelige_chunks.append(chunk)
        else:
            endelige_chunks.extend(_del_lang_tekst(chunk, maks_lengde, overlapp))

    return endelige_chunks if endelige_chunks else [tekst]

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
    print(f"[Database] Indeksert {filnavn} ({len(chunks)} chunks/regler).")

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
        if not os.path.exists(mappesti):
            os.makedirs(mappesti)
        
        for rot, _, filer in os.walk(mappesti):
            for filnavn in filer:
                oppdater_fil_i_database(os.path.join(rot, filnavn))

        observer.schedule(handler, path=mappesti, recursive=True)

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
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    
    filtrerte_kilder = []
    for i in range(len(dokumenter)):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        utdrag_tekst = f"Fil: {kilde} (Del {chunk_nr})\nInnhold:\n{dokumenter[i]}"
        if utdrag_tekst not in filtrerte_kilder and len(filtrerte_kilder) < 5:
            filtrerte_kilder.append(utdrag_tekst)

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    meldinger = [
        {
            "role": "system",
            "content": (
                "Du er en konsis dokumentassistent. Du svarer utelukkende basert på de oppgitte kildene, med naturlig, flytende norsk språk.\n\n"
                "Du skal alltid prioritere informasjonen i de oppgitte kildene. Hvis du er i tvil om informasjonen finnes, gå grundig gjennom alle kilde-utdragene en ekstra gang før du konkluderer. "
                "Svar direkte, korrekt og uten unødvendige høflighetsfraser. "
                
                "REGLER FOR SVAR:\n"
                "1. Svar direkte på det som blir spurt om, uten å kommentere spørsmålsstillingen, brukerens antakelser eller hvorfor de spør.\n"
                "2. Hvis brukeren oppgir en verdi eller påstand som ikke stemmer med kildene, skal du korrigere dette naturlig i svaret – bekreft at det er feil, og gi deretter den riktige informasjonen i samme setning, i stedet for å bare svare ja eller nei.\n"
                "3. Hvis kildene ikke inneholder svaret, si tydelig at informasjonen ikke finnes i kildene du har tilgjengelig. Ikke gjett eller fyll inn med generell kunnskap.\n"
                "4. Konkrete verdier som datoer, klokkeslett, tall og navn skal alltid gjengis EKSAKT slik de står i kildeteksten. Ikke rund av, ikke generaliser, og ikke legg til årstall, klokkeslett eller andre detaljer som ikke eksplisitt står i kilden.\n"
                "5. Hvis du er usikker på om en verdi stemmer med kildene, skal du heller si at du er usikker enn å gjette.\n"
                "6. Ikke bruk faste innledningsfraser, maler eller unødvendige høflighetsfraser (f.eks. 'Takk for spørsmålet', 'Basert på kildene kan jeg fortelle at...'). Gå rett på svaret.\n"
                "7. Hold svarene korte og presise – ikke lengre enn nødvendig for å svare fullstendig.\n"
                "8. Hvis spørsmålet er tvetydig og kildene inneholder flere mulige svar, spesifiser hvilket du svarer ut ifra, uten å gjette på hva brukeren mente.\n"
                "132. IKKE dikt opp datoer, årstall eller verdier som ikke står eksplisitt i kilden.\n"
                "332. Hvis informasjonen ikke finnes, si klart at informasjonen mangler i kildene og ikke noe annet. Ikke gjett.\n"
                "432. Svar direkte og korrekt uten unødvendige høflighetsfraser."
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
    
    print(f"-> Svar fra AI:\n{svar}\n")

if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    forste_indeksering_ferdig.wait(timeout=60)

    print("\n[System] Klar til å ta imot spørsmål i terminalen!")
    while True:
        try:
            sporsmal = input("Hva vil du spørre om? (eller 'avslutt'): ").strip()
            if sporsmal.lower() in ("avslutt", "exit", "quit"): 
                break
            if not sporsmal:
                continue
            
            if sporsmal.lower().startswith("hvor finnes ordet"):
                ordet = sporsmal.replace("hvor finnes ordet", "").replace("'", "").strip()
                sok_etter_ord_i_filer(ordet)
            else:
                besvar_sporsmal(sporsmal)
        except KeyboardInterrupt: 
            break