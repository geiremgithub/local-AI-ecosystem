import os
import sys
import warnings
import time
import json
from threading import Thread, Event

# --- Oppsett for norsk tekst i terminal ---
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v43-agentisk-sok"
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
    max_new_tokens=250,
    do_sample=True,
    temperature=0.1,
)

# --- AGENT-VERKTØY: Eksakt tekstsøk på tvers av alle mapper ---
def søk_eksakt_tekst(sokefrase):
    """Skanner alle mapper og undermapper etter en eksakt frasetekst, og henter ut avsnittene."""
    print(f"\n[Agent-Verktøy] Kjører eksakt søk etter: '{sokefrase}'")
    resultater = []
    
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
                        # Sjekker om frasen finnes (uavhengig av store/små bokstaver)
                        if sokefrase.lower() in innhold.lower():
                            # Hent ut linjer eller avsnitt som inneholder frasen
                             avsnitt = [avs.strip() for avs in innhold.split("\n\n") if sokefrase.lower() in avs.lower()]
                             if not avsnitt:
                                 avsnitt = [innhold[:300]] # Standard utdrag hvis ikke delt i avsnitt
                            resultater.append({
                                "filnavn": filnavn,
                                "sti": full_sti,
                                "utdrag": avsnitt
                            })
                except Exception as e:
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
            
            vektor_snutt = [round(float(v), 2) for v in dokument_vektorer[i][:4]]
            vektor_str = f"Vektor: {vektor_snutt}..."
            
            etikett = f"[{kilde} - C{chunk_nr}]\n\"{smakebit}\"\n{vektor_str}\n(Avst: {dist:.4f})"
            plt.scatter(reduserte[i+1][0], reduserte[i+1][1], s=120, alpha=0.8)
            plt.text(reduserte[i+1][0] + 0.02, reduserte[i+1][1], etikett, fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.75, boxstyle='round,pad=0.3'))
            
        sporsmal_vektor_snutt = [round(float(v), 2) for v in sporsmal_vektor[:4]]
        sporsmal_vektor_str = f"Vektor: {sporsmal_vektor_snutt}..."
        
        plt.scatter(reduserte[0][0], reduserte[0][1], color='red', s=300, marker='*', zorder=5, label='Spørsmål')
        sporsmal_etikett = f"Spørsmål:\n\"{bruker_sporsmal}\"\n{sporsmal_vektor_str}"
        plt.text(reduserte[0][0] + 0.02, reduserte[0][1], sporsmal_etikett, fontsize=9, fontweight='bold', color='red',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'), zorder=6)
        
        plt.title("Vektorrom-visualisering av RAG-søk", fontsize=14, fontweight='bold')
        plt.xlabel("PCA Dimensjon 1")
        plt.ylabel("PCA Dimensjon 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        
        grunnavn = "vektorrom_visualisering"
        filtype = ".png"
        utfil_sti = f"{grunnavn}{filtype}"
        teller = 1
        
        while os.path.exists(utfil_sti):
            utfil_sti = f"{grunnavn}({teller}){filtype}"
            teller += 1
            
        plt.savefig(utfil_sti, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Grafikk] Lagret bildefil: {utfil_sti}")
    except Exception as e:
        print(f"[Feil] Kunne ikke generere grafikk: {e}")

# --- Global liste for chathistorikk ---
chathistorikk = []

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG & AGENT PROSESSPESIFIKASJON] ---")
    
    # 1. Standard RAG databasesøk (vektorer)
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

    # Sjekk om brukeren ber om et spesifikt tekstsøk / kryssoppslag (Agent-logikk)
    agent_ekstra_info = ""
    if any(ord i bruker_sporsmal.lower() for ord in ["finn", "hvilke filer", "eksakt", "sammenlign"]):
        # Hent ut et nøkkelord eller bruk hele spørsmålet som søkefrase
        sokefrase = bruker_sporsmal.replace("finn", "").replace("hvilke filer", "").replace("som", "").strip()
        if len(sokefrase) > 3:
            agent_resultat = søk_eksakt_tekst(sokefrase)
            agent_ekstra_info = f"\n\n----- RESULTATER FRA EKSAKT VERKTØYSØK -----\n{agent_resultat}"

    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en presis assistent som snakker flytende og korrekt norsk. "
                "Svar direkte på brukerens spørsmål basert på oppgitt dokumentinnhold og verktøyresultater. "
                "Hvis verktøyresultater viser flere filer med lik eller varierende tekst, skal du opprette en punktliste "
                "over hvem som skrev hva i de ulike filene.\n\n"
                f"----- DOKUMENTINNHOLD (RAG) -----\n{kombinert_kontekst}"
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
    
    antall_ord = len(svar.split())
    antall_tokens_omtrent = int(antall_ord * 1.3)
    hastighet = antall_tokens_omtrent / total_tid if total_tid > 0 else 0

    print(f"-> Svar fra AI:\n{svar}")
    print(f"[Ytelse] Generering tok {total_tid:.2f} sekunder (ca. {hastighet:.1f} tokens/sek)\n")

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