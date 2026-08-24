import os
import sys
import warnings
import time
import json
import re
from threading import Thread, Event

# --- Oppsett for norsk tekst i terminal ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v47-agentisk-sok-telling-fix"
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

# --- Biblioteker for visualisering ---
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

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

tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80,
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

# --- AGENT-VERKTØY: Nøyaktig telling og tekstsøk ---
def søk_eksakt_tekst(sokefrase):
    print(f"\n[Agent-Verktøy] Kjører nøyaktig telling og søk etter: '{sokefrase}'")
    resultater = []
    
    try:
        mønster = re.compile(re.escape(sokefrase), re.IGNORECASE)
    except Exception:
        return json.dumps([], ensure_ascii=False)

    for mappe in MAPPESTIER:
        if not os.path.exists(mappe):
            continue
        for rot, _, filer in os.walk(mappe):
            for filnavn in filer:
                if not filnavn.endswith(".txt"):
                    continue
                full_sti = os.path.join(rot, filnavn)
                try:
                    with open(full_sti, "r", encoding="utf-8") as f:
                        innhold = f.read()
                        alle_linjer = innhold.split("\n")
                        treff_linjer = [linje.strip() for linje in alle_linjer if mønster.search(linje)]
                        
                        if treff_linjer:
                            resultater.append({
                                "filnavn": filnavn,
                                "sti": full_sti,
                                "antall_treff": len(treff_linjer),
                                "utdrag": treff_linjer
                            })
                except Exception:
                    pass
                    
    return json.dumps(resultater, ensure_ascii=False)

def generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander, sporsmal_vektor, dokument_vektorer):
    print("[Grafikk] Genererer vektorrom-visualisering...")
    try:
        alle_vektorer = np.array([sporsmal_vektor] + list(dokument_vektorer))
        pca = PCA(n_components=2)
        reduserte = pca.fit_transform(alle_vektorer)
        
        plt.figure(figsize=(14, 10))
        for i in range(len(dokumenter)):
            kilde = metadataer[i].get("kilde", "ukjent")
            chunk_nr = metadataer[i].get("chunk_nr", 0)
            smakebit = dokumenter[i][:35].replace("\n", " ") + "..."
            dist = avstander[i]
            
            etikett = f"[{kilde} - C{chunk_nr}]\n\"{smakebit}\"\n(Avst: {dist:.4f})"
            plt.scatter(reduserte[i+1][0], reduserte[i+1][1], s=120, alpha=0.8)
            plt.text(reduserte[i+1][0] + 0.02, reduserte[i+1][1], etikett, fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.75, boxstyle='round,pad=0.3'))
            
        plt.scatter(reduserte[0][0], reduserte[0][1], color='red', s=300, marker='*', zorder=5, label='Spørsmål')
        plt.title("Vektorrom-visualisering av RAG-søk", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        
        grunnavn = "vektorrom_visualisering"
        utfil_sti = f"{grunnavn}.png"
        teller = 1
        while os.path.exists(utfil_sti):
            utfil_sti = f"{grunnavn}({teller}).png"
            teller += 1
            
        plt.savefig(utfil_sti, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Grafikk] Lagret bildefil: {utfil_sti}")
    except Exception as e:
        print(f"[Feil] Kunne ikke generere grafikk: {e}")

chathistorikk = []

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG & AGENT PROSESSPESIFIKASJON] ---")
    
    sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0] if "distances" in sok_resultat else [0.0] * len(dokumenter)

    dok_vektorer = embedding_modell(dokumenter)
    generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander, sporsmal_vektor, dok_vektorer)

    kombinert_kontekst = "\n\n".join(dokumenter[:5])

    agent_ekstra_info = ""
    if any(ord in bruker_sporsmal.lower() for ord in ["finn", "hvilke filer", "eksakt", "sammenlign"]):
        match = re.search(r"['\"](.*?)['\"]", bruker_sporsmal)
        sokefrase = match.group(1) if match else "skolestart"
            
        agent_resultat_json = søk_eksakt_tekst(sokefrase)
        
        try:
            parset_resultat = json.loads(agent_resultat_json)
            formatert_agent_tekst = f"Søkefrase: '{sokefrase}'\n\n"
            for res in parset_resultat:
                formatert_agent_tekst += f"FILNAVN: {res['filnavn']}\n"
                formatert_agent_tekst += f"ANTALL GANGER ORDET FOREKOMMER: {res['antall_treff']}\n"
                formatert_agent_tekst += "Linjer:\n"
                for utd in res['utdrag']:
                    formatert_agent_tekst += f"  - {utd}\n"
                formatert_agent_tekst += "\n"
        except Exception:
            formatert_agent_tekst = agent_resultat_json

        agent_ekstra_info = f"\n\n----- FASIT FRA VERKTØYET (BRUK KUN DENNE INFORMASJONEN) -----\n{formatert_agent_tekst}"

    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en streng og nøyaktig dataassistent. "
                "Du skal basere deg 100% på 'FASIT FRA VERKTØYET' når det gjelder antall forekomster og linjer per fil. "
                "Ikke finn på eller anta at filer har samme innhold hvis fasiten viser ulikheter. "
                "Vis tydelig hva som varierer mellom filene basert på antall treff og faktisk innhold.\n\n"
                f"----- RAG-KONTEKST -----\n{kombinert_kontekst}"
                f"{agent_ekstra_info}"
            )
        },
    ]
    
    meldinger.extend(chathistorikk)
    meldinger.append({"role": "user", "content": bruker_sporsmal})

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)

    print("4. LLM-Generering: Sender prompt til Qwen-modellen...")
    start_tid = time.time()
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    slutt_tid = time.time()
    
    total_tid = slutt_tid - start_tid
    svar = resultat[0]["generated_text"].strip()
    
    print(f"-> Svar fra AI:\n{svar}")
    print(f"[Ytelse] Generering tok {total_tid:.2f} sekunder\n")

    chathistorikk.append({"role": "user", "content": bruker_sporsmal})
    chathistorikk.append({"role": "assistant", "content": svar})

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