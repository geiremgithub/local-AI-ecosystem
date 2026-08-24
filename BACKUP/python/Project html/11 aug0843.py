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

SKRIPT_VERSJON = "v17-terminal - med chunks-debug"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16
    print(f"[System] GPU funnet ({torch.cuda.get_device_name(0)}) - bruker CUDA for rask inferens.")
else:
    ENHET = "cpu"
    PRESISJON = torch.float32
    torch.set_num_threads(os.cpu_count())
    print(f"[System] Ingen GPU funnet - kjører på CPU med {os.cpu_count()} tråder.")

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
        print(f"[Database] Fjernet {filnavn} (filen ble slettet).")
        return

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
    except UnicodeDecodeError:
        try:
            with open(full_sti, "r", encoding="cp1252") as fil:
                tekst = fil.read().strip()
        except Exception as feil:
            print(f"[Feil] Klarte ikke å lese {filnavn}: {feil}")
            return
    except Exception as feil:
        print(f"[Feil] Uventet feil ved lesing av {filnavn}: {feil}")
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
    print(f"[Database] Suksess! Delte {filnavn} inn i {len(chunks)} biter med overlapping.")

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

forste_indeksering_ferdig = Event()

def start_mappe_overvaking():
    try:
        print("[System] Gjør første skanning og oppretter biter...")
        if not os.path.exists(MAPPESTI):
            os.makedirs(MAPPESTI)
        for filnavn in os.listdir(MAPPESTI):
            oppdater_fil_i_database(os.path.join(MAPPESTI, filnavn))

        handler = DokumentLytter()
        observer = Observer()
        observer.schedule(handler, path=MAPPESTI, recursive=False)
        observer.start()
        print(f"[Watchdog] Lytter aktivt i: {MAPPESTI}")
    except Exception as feil:
        print(f"[Feil] Noe gikk galt under oppstart: {feil}")
    finally:
        forste_indeksering_ferdig.set()

print("[System] Laster inn Qwen-modellen...")
ai_modell = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device=ENHET,
    torch_dtype=PRESISJON,
)

generasjons_konfig = GenerationConfig(
    max_new_tokens=180,
    max_length=None,
    do_sample=False,
)

MODUS_INSTRUKSER = {
    "høflig": "Formuler SVAR-linjen formelt og høflig.",
}
STANDARD_MODUS = "høflig"
SAMTALE_HISTORIKK = []
MAKS_HISTORIKK = 3

def bygg_sok_sporsmal(bruker_sporsmal):
    if not SAMTALE_HISTORIKK:
        return bruker_sporsmal
    forrige = SAMTALE_HISTORIKK[-1]["sporsmal"]
    return f"{forrige} {bruker_sporsmal}"

def besvar_sporsmal(bruker_sporsmal):
    print("Søker etter mest relevante tekst-biter...")
    sok_sporsmal = bygg_sok_sporsmal(bruker_sporsmal)
    sok_resultat = db_samling.query(query_texts=[sok_sporsmal], n_results=8)

    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar fra AI: Fant ingen relevante dokumenter.\n")
        return

    dokument_kontekster = sok_resultat["documents"][0]
    metadata_liste = sok_resultat["metadatas"][0]
    avstander = sok_resultat.get("distances", [[None] * len(dokument_kontekster)])[0]

    sporsmal_ord = set(w.lower().strip(".,?!") for w in sok_sporsmal.split() if len(w) > 3)

    def nokkelord_score(tekst):
        tekst_ord = set(w.lower().strip(".,?!") for w in tekst.split())
        return len(sporsmal_ord & tekst_ord)

    kandidater = []
    for tekst_bit, meta, avstand in zip(dokument_kontekster, metadata_liste, avstander):
        kandidater.append((tekst_bit, meta, avstand, nokkelord_score(tekst_bit)))

    kandidater.sort(key=lambda x: (-x[3], x[2] if x[2] is not None else float("inf")))

    print("Debug - hentede biter (sortert etter nøkkelord + avstand):")
    for i, (tekst_bit, meta, avstand, score) in enumerate(kandidater):
        avstand_str = f"{avstand:.4f}" if avstand is not None else "None"
        print(f"  [{i}] kilde={meta.get('kilde')} chunk={meta.get('chunk_nr')} avstand={avstand_str} nøkkelord={score} | \"{tekst_bit[:60]}...\"")

    beste_kandidater = kandidater[:5]
    dokument_kontekster = [k[0] for k in beste_kandidater]
    metadata_liste = [k[1] for k in beste_kandidater]

    print(f"AI-en analyserer {len(dokument_kontekster)} bit(er), best treff fra: {metadata_liste[0].get('kilde')}...")

    kombinert_kontekst = "\n\n".join(
        f"TEKSTBIT {i + 1}:\n{tekst}" for i, tekst in enumerate(dokument_kontekster)
    )
    
    meldinger = [
        {
            "role": "system",
            "content": (
                "Du er en presis assistent. Svar ALLTID med en linje som starter med: "
                "SVAR: <kort svar på norsk, én setning>\n\n"
                f"----- EKTE DOKUMENTINNHOLD -----\n{kombinert_kontekst}"
            ),
        },
    ]

    for tur in SAMTALE_HISTORIKK[-MAKS_HISTORIKK:]:
        meldinger.append({"role": "user", "content": tur["sporsmal"]})
        meldinger.append({"role": "assistant", "content": f"SVAR: {tur['svar']}"})

    meldinger.append({"role": "user", "content": bruker_sporsmal})

    prompt = ai_modell.tokenizer.apply_chat_template(
        meldinger, tokenize=False, add_generation_prompt=True
    )

    resultat = ai_modell(
        prompt,
        generation_config=generasjons_konfig,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    full_generert_tekst = resultat[0]["generated_text"]
    svar = ""
    for linje in full_generert_tekst.split("\n"):
        linje = linje.strip()
        if linje.upper().startswith("SVAR:"):
            svar = linje.split(":", 1)[1].strip()

    if not svar:
        svar = full_generert_tekst.strip()
    if svar and not svar.endswith("."):
        svar += "."

    def finn_kilde_for_svar(svar_tekst, kontekst_biter, metadata):
        svar_ord = set(w.lower().strip(".,?!") for w in svar_tekst.split() if len(w) > 3)
        beste_indeks, beste_overlapp = 0, -1
        for i, bit in enumerate(kontekst_biter):
            bit_ord = set(w.lower().strip(".,?!") for w in bit.split())
            overlapp = len(svar_ord & bit_ord)
            if overlapp > beste_overlapp:
                beste_overlapp, beste_indeks = overlapp, i
        return metadata[beste_indeks]["kilde"], metadata[beste_indeks]["chunk_nr"]

    kilde_fil, chunk_nr = finn_kilde_for_svar(svar, dokument_kontekster, metadata_liste)

    print(f"-> Svar fra AI: [{kilde_fil} | Chunk {chunk_nr}] {svar}\n")

    SAMTALE_HISTORIKK.append({"sporsmal": bruker_sporsmal, "svar": svar})
    del SAMTALE_HISTORIKK[:-MAKS_HISTORIKK]

if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    forste_indeksering_ferdig.wait(timeout=60)

    print("\n[System] Klar til å ta imot spørsmål i terminalen!")
    while True:
        try:
            bruker_sporsmal = input("Hva vil du spørre om? (eller skriv 'avslutt'): ").strip()
            if bruker_sporsmal.lower() in ("avslutt", "exit", "quit"):
                break
            if not bruker_sporsmal:
                continue
            besvar_sporsmal(bruker_sporsmal)
        except KeyboardInterrupt:
            break