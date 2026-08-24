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

SKRIPT_VERSJON = "v2c-full-rag-flask-mcp-komplett"
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

# --- Flask og MCP ---
from flask import Flask, request, jsonify, render_template_string
from mcp.server.fastmcp import FastMCP

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

tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

# --- Global samtalehistorikk og Flask-logg ---
SAMTALE_HISTORIKK = []
MAKS_HISTORIKK_MELDINGER = 6 
FLASK_HISTORIKK_LOGG = []

def del_tekst_i_chunks(tekst):
    return tekst_splitter.split_text(tekst)

def _unik_kilde_navn(full_sti):
    for mappesti in MAPPESTIER:
        try:
            rel = os.path.relpath(full_sti, mappesti)
        except ValueError:
            continue
        if not rel.startswith(".."):
            mappe_navn = os.path.basename(os.path.normpath(mappesti))
            return f"{mappe_navn}/{rel.replace(os.sep, '/')}"
    return os.path.basename(full_sti)

def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"):
        return
    filnavn = _unik_kilde_navn(full_sti)
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
        try:
            with open(full_sti, "r", encoding="cp1252") as fil:
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
    repetition_penalty=1.1,
)

def besvar_sporsmal(bruker_sporsmal):
    global SAMTALE_HISTORIKK

    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        svar = "Fant ingen relevante dokumenter."
        print(f"-> Svar: {svar}\n")
        return svar

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]

    filtrerte_kilder = []
    for i in range(len(dokumenter)):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        utdrag_tekst = f"Fil: {kilde} (Del {chunk_nr})\nInnhold:\n{dokumenter[i]}"
        if utdrag_tekst not in filtrerte_kilder:
            filtrerte_kilder.append(utdrag_tekst)

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    meldinger = [
        {
            "role": "system",
            "content": (
                "1. Du er en konsis dokumentassistent. "
                "2. Svar utelukkende basert på de oppgitte kildene. "
                "3. Du skal alltid prioritere informasjonen i de oppgitte kildene. "
                "4. Hvis du er i tvil om informasjonen finnes, gå grundig gjennom alle kilde-utdragene en ekstra gang før du konkluderer. "
                "5. Svar direkte, korrekt og uten unødvendige høflighetsfraser, med naturlig, flytende norsk språk.\n\n"
                "REGLER FOR SVAR:\n"
                "6. Svar direkte på det som blir spurt om, uten å kommentere spørsmålsstillingen, brukerens antakelser eller hvorfor de spør.\n"
                "7. Hvis brukeren oppgir en verdi eller påstand som skal bekreftes eller avkreftes mot kildene, skal du ALLTID starte svaret med et eksplisitt 'Nei,' (hvis påstanden er feil) eller 'Ja,' (hvis påstanden stemmer), og deretter oppgi den korrekte informasjonen fra kilden i samme setning. Ikke bare korriger implisitt - det eksplisitte Ja/Nei skal alltid stå først.\n"
                "8. Hvis kildene ikke inneholder svaret, si tydelig at informasjonen ikke finnes i kildene du har tilgjengelig. "
                "9. Ikke gjett eller fyll inn med generell kunnskap. "
                "10. Konkrete verdier som datoer, klokkeslett, tall og navn skal alltid gjengis EKSAKT slik de står i kildeteksten. "
                "11. Ikke rund av, ikke generaliser, og ikke legg til årstall, klokkeslett eller andre detaljer som ikke eksplisitt står i kilden. "
                "12. Hvis du er usikker på om en verdi stemmer med kildene, skal du heller si at du er usikker enn å gjette. "
                "13. Ikke bruk faste innledningsfraser, maler eller unødvendige høflighetsfraser. "
                "14. Gå rett på svaret. "
                "15. Hold svarene korte og presise – ikke lengre enn nødvendig for å svare fullstendig. "
                "16. Hvis spørsmålet er tvetydig og kildene inneholder flere mulige svar, spesifiser hvilket du svarer ut ifra, uten å gjette på hva brukeren mente. "
                "17. Ikke kom med selvmotsigende konklusjoner eller ugyldige logiske slutninger. "
                "18. Ikke dikt opp datoer, årstall eller verdier som ikke står eksplisitt i kilden. "
                "19. Hvis informasjonen ikke finnes, si klart at informasjonen mangler i kildene. "
                "20. Ikke gjett. "
                "21. Svar direkte og korrekt uten unødvendige høflighetsfraser."
            )
        }
    ]

    meldinger.extend(SAMTALE_HISTORIKK)
    meldinger.append({
        "role": "user",
        "content": f"Spørsmål: {bruker_sporsmal}\n\nRelevante kilder:\n{kombinert_kontekst}"
    })

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    svar = resultat[0]["generated_text"].strip()

    SAMTALE_HISTORIKK.append({"role": "user", "content": bruker_sporsmal})
    SAMTALE_HISTORIKK.append({"role": "assistant", "content": svar})
    if len(SAMTALE_HISTORIKK) > MAKS_HISTORIKK_MELDINGER:
        SAMTALE_HISTORIKK = SAMTALE_HISTORIKK[-MAKS_HISTORIKK_MELDINGER:]

    FLASK_HISTORIKK_LOGG.insert(0, {
        "tid": time.strftime("%H:%M:%S"),
        "sporsmal": bruker_sporsmal,
        "svar": svar,
        "kilde": metadataer[0].get("kilde", "ukjent"),
        "chunk": metadataer[0].get("chunk_nr", 0)
    })

    print(f"\n-> Svar fra AI:\n{svar}\n")
    return svar

# --- Flask Oppsett ---
app = Flask(__name__)

HTML_VISNING = """
<!DOCTYPE html>
<html lang="no">
<head>
    <meta charset="UTF-8">
    <title>RAG Systemlogg</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f4f4f9; }
        table { width: 100%; border-collapse: collapse; background: white; margin-top: 15px;}
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #eee; }
    </style>
</head>
<body>
    <h1>RAG Systemlogg & Loggvisning</h1>
    <table>
        <tr><th>Tid</th><th>Spørsmål</th><th>Svar</th><th>Kilde</th></tr>
        {% for item in logg %}
        <tr>
            <td>{{ item.tid }}</td>
            <td>{{ item.sporsmal }}</td>
            <td>{{ item.svar }}</td>
            <td>{{ item.kilde }} (Chunk {{ item.chunk }})</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_VISNING, logg=FLASK_HISTORIKK_LOGG)

@app.route("/api/sporsmal", methods=["POST"])
def api_sporsmal():
    data = request.get_json(force=True, silent=True) or {}
    spm = data.get("sporsmal", "").strip()
    if not spm:
        return jsonify({"feil": "Tomt spørsmål"}), 400
    svar = besvar_sporsmal(spm)
    return jsonify({"svar": svar})

def kjør_flask():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# --- MCP Server Oppsett ---
mcp = FastMCP("dokumentoppslag")

@mcp.tool()
def sok_i_dokumenter(sporsmal: str, antall_treff: int = 5) -> str:
    """Søk i den lokale dokumentmappen og hent ut relevante tekstbiter."""
    sok_resultat = db_samling.query(query_texts=[sporsmal], n_results=antall_treff)
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        return "Ingen relevante dokumenter funnet."
    
    deler = []
    for i, (tekst, meta) in enumerate(zip(sok_resultat["documents"][0], sok_resultat["metadatas"][0]), 1):
        deler.append(f"TREFF {i} (kilde: {meta['kilde']}, bit {meta['chunk_nr'}):\n{tekst}")
    return "\n\n".join(deler)

@mcp.tool()
def list_dokumenter() -> str:
    """List opp alle indekserte dokumentfiler."""
    alle = db_samling.get(include=["metadatas"])
    kilder = sorted(set(m["kilde"] for m in alle["metadatas"]))
    return "\n".join(kilder) if kilder else "Ingen dokumenter indeksert."

if __name__ == "__main__":
    # Start mappeovervåking i bakgrunnen
    Thread(target=start_mappe_overvaking, daemon=True).start()
    forste_indeksering_ferdig.wait(timeout=60)

    # Start Flask-webserver i bakgrunnen
    Thread(target=kjør_flask, daemon=True).start()
    print("[System] Flask web-logg kjører på http://127.0.0.1:5000")

    # Start MCP-serveren på stdio
    print("[System] Starter MCP-server...")
    mcp.run(transport="stdio")