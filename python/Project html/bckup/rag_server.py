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

SKRIPT_VERSJON = "v17-web - med terminal-debug for chunks"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

# ---------------------------------------------------------
# Samme enhet/presisjon-logikk som i terminal-versjonen
# ---------------------------------------------------------
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

from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# Stier til mapper - juster disse til dine egne
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
        print(f"[Advarsel] {filnavn} er ikke UTF-8-kodet, prøver Windows-1252 (ANSI) i stedet...")
        try:
            with open(full_sti, "r", encoding="cp1252") as fil:
                tekst = fil.read().strip()
        except Exception as feil:
            print(f"[Feil] Klarte ikke å lese {filnavn} uansett tegnsett, hopper over: {feil}")
            return
    except Exception as feil:
        print(f"[Feil] Uventet feil ved lesing av {filnavn}, hopper over: {feil}")
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
        print(f"[Feil] Noe gikk galt under oppstart av mappeovervåking: {feil}")
        forste_indeksering_ferdig.set()
        return

    forste_indeksering_ferdig.set()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ---------------------------------------------------------
# 3. Qwen (generator-modellen)
# ---------------------------------------------------------
print("[System] Laster inn Qwen-modellen... (dette tar litt tid, skjer kun ved oppstart)")

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
    "høflig": (
        "Formuler SVAR-linjen formelt og høflig, med fullstendige setninger "
        "og et respektfullt språk."
    ),
    "morsom": (
        "Formuler SVAR-linjen på en tydelig munter og humoristisk måte - bruk "
        "gjerne et vidd, en morsom sammenligning eller et lett overdrevet "
        "uttrykk."
    ),
    "kort": (
        "Formuler SVAR-linjen så kort og direkte som mulig, uten "
        "unødvendige ord - men behold all viktig informasjon fra FAKTA."
    ),
    "vennlig": (
        "Formuler SVAR-linjen uformelt og vennlig, som om du forklarer det "
        "til en god venn."
    ),
}
STANDARD_MODUS = "høflig"

SAMTALE_HISTORIKK = []
MAKS_HISTORIKK = 3


def bygg_sok_sporsmal(bruker_sporsmal):
    if not SAMTALE_HISTORIKK:
        return bruker_sporsmal
    forrige = SAMTALE_HISTORIKK[-1]["sporsmal"]
    return f"{forrige} {bruker_sporsmal}"


# ---------------------------------------------------------
# 4. Spørre-funksjon med terminal-debug av chunks
# ---------------------------------------------------------
def besvar_sporsmal(bruker_sporsmal, gjeldende_modus):
    if gjeldende_modus not in MODUS_INSTRUKSER:
        gjeldende_modus = STANDARD_MODUS

    print("Søker etter mest relevante tekst-biter...")
    sok_sporsmal = bygg_sok_sporsmal(bruker_sporsmal)
    sok_resultat = db_samling.query(query_texts=[sok_sporsmal], n_results=8)

    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        return {"svar": "Fant ingen relevante dokumenter.", "fakta": "", "sammenligning": "",
                "kilde": "", "chunk": None, "advarsel": None, "kandidater": []}

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
    undersokte_biter = []
    for i, (tekst_bit, meta, avstand, score) in enumerate(kandidater):
        avstand_str = f"{avstand:.4f}" if avstand is not None else "None"
        print(f"  [{i}] kilde={meta.get('kilde')} chunk={meta.get('chunk_nr')} avstand={avstand_str} nøkkelord={score} | \"{tekst_bit[:60]}...\"")
        undersokte_biter.append({
            "kilde": meta.get("kilde"),
            "chunk": meta.get("chunk_nr"),
            "avstand": avstand,
            "nokkelord": score,
            "tekst": tekst_bit
        })

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
                "Du er en presis assistent. Svar ALLTID i nøyaktig dette "
                "formatet, med tre linjer:\n"
                "FAKTA: <den setningen fra tekstbitene under som er mest "
                "relevant for spørsmålet/påstanden>\n"
                "SAMMENLIGNING: <kun hvis brukeren skrev en PÅSTAND... skriv 'ikke aktuelt - rent spørsmål'>\n"
                "SVAR: <kort svar på norsk, én setning>\n\n"
                f"----- EKTE DOKUMENTINNHOLD -----\n{kombinert_kontekst}\n\n"
                f"TONE FOR SVAR-LINJEN: {MODUS_INSTRUKSER[gjeldende_modus]}"
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
    svar_linjer = full_generert_tekst.split("\n")

    svar, fakta_linje, sammenligning_linje = "", "", ""
    for linje in svar_linjer:
        linje = linje.strip()
        if linje.upper().startswith("SVAR:"):
            svar = linje.split(":", 1)[1].strip()
        elif linje.upper().startswith("FAKTA:"):
            fakta_linje = linje.split(":", 1)[1].strip()
        elif linje.upper().startswith("SAMMENLIGNING:"):
            sammenligning_linje = linje.split(":", 1)[1].strip()

    if not svar:
        svar = full_generert_tekst.strip()
    if svar and not svar.endswith("."):
        svar += "."

    def finn_kilde_for_fakta(fakta, kontekst_biter, metadata):
        if not fakta or fakta.strip().lower() in ("(ingen)", "ingen", ""):
            return metadata[0]["kilde"], metadata[0]["chunk_nr"]
        fakta_ord = set(w.lower().strip(".,?!") for w in fakta.split() if len(w) > 3)
        beste_indeks, beste_overlapp = 0, -1
        for i, bit in enumerate(kontekst_biter):
            bit_ord = set(w.lower().strip(".,?!") for w in bit.split())
            overlapp = len(fakta_ord & bit_ord)
            if overlapp > beste_overlapp:
                beste_overlapp, beste_indeks = overlapp, i
        return metadata[beste_indeks]["kilde"], metadata[beste_indeks]["chunk_nr"]

    kilde_fil, chunk_nr = finn_kilde_for_fakta(fakta_linje, dokument_kontekster, metadata_liste)

    print(f"-> Svar fra AI: [{kilde_fil} | Chunk {chunk_nr}] {svar}\n")

    SAMTALE_HISTORIKK.append({"sporsmal": bruker_sporsmal, "svar": svar})
    del SAMTALE_HISTORIKK[:-MAKS_HISTORIKK]

    return {
        "svar": svar,
        "fakta": fakta_linje,
        "sammenligning": sammenligning_linje,
        "kilde": kilde_fil,
        "chunk": chunk_nr,
        "advarsel": None,
        "kandidater": undersokte_biter
    }


# ---------------------------------------------------------
# 5. Flask-appen
# ---------------------------------------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html", moduser=list(MODUS_INSTRUKSER.keys()), standard_modus=STANDARD_MODUS)


@app.route("/api/sporsmal", methods=["POST"])
def api_sporsmal():
    data = request.get_json(force=True, silent=True) or {}
    bruker_sporsmal = (data.get("sporsmal") or "").strip()
    gjeldende_modus = data.get("modus") or STANDARD_MODUS

    if not bruker_sporsmal:
        return jsonify({"feil": "Tomt spørsmål."}), 400

    try:
        resultat = besvar_sporsmal(bruker_sporsmal, gjeldende_modus)
        return jsonify(resultat)
    except Exception as feil:
        print(f"[Feil] Uventet feil under besvarelse: {feil}")
        return jsonify({"feil": f"Noe gikk galt på serveren: {feil}"}), 500


if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    if not forste_indeksering_ferdig.wait(timeout=60):
        print("[Advarsel] Indekseringen brukte uventet lang tid (>60s).")

    print("[System] Serveren er klar. Åpne http://127.0.0.1:5000 i nettleseren din.")
    app.run(host="127.0.0.1", port=5000, debug=False)