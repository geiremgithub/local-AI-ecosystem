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

# --- Biblioteker for visualisering ---
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

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

def del_tekst_i_chunks(tekst):
    enkelte_regler = re.split(r'\n(?=\d+\.\s)', tekst.strip())
    chunks = []
    for regel in enkelte_regler:
        if regel.strip():
            chunks.append(regel.strip())
    
    if not chunks or len(chunks) == 1:
        return [tekst]
    return chunks

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

def generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander, sporsmal_vektor, dokument_vektorer):
    print("[Grafikk] Genererer vektorrom-visualisering (lagres til fil: vektorrom_visualisering.png)...")
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
        
        plt.savefig("vektorrom_visualisering.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[Grafikk] Lagret bildefil: vektorrom_visualisering.png")
    except Exception as e:
        print(f"[Feil] Kunne ikke generere grafikk: {e}")

def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    sporsmal_tokens = ai_modell.tokenizer.tokenize(bruker_sporsmal)
    fiksede_tokens = [t.encode('latin1', errors='ignore').decode('utf-8', errors='ignore') for t in sporsmal_tokens]
    sporsmal_token_ids = ai_modell.tokenizer.encode(bruker_sporsmal)
    
    print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(sporsmal_tokens)} tokens.")
    print(f"   - Tokens: {fiksede_tokens}")
    print(f"   - Token ID-er: {sporsmal_token_ids}")

    sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
    formaterte_tall_sporsmal = [round(float(v), 4) for v in sporsmal_vektor[:8]]
    print(f"2. Embedding (Spørsmål-vektor): Totalt {len(sporsmal_vektor)} dimensjoner.")
    print(f"   - Vektortall [format]: {formaterte_tall_sporsmal} ...")

    print("3. Databasesøk...")
    sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=5)
    
    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar: Fant ingen relevante dokumenter.\n")
        return

    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0] if "distances" in sok_resultat else [0.0] * len(dokumenter)

    print("   - Topp treff fra databasen og deres vektorer:")
    dok_vektorer = embedding_modell(dokumenter)
    
    filtrerte_kilder = []
    
    for i in range(len(dokumenter)):
        kilde = metadataer[i].get("kilde", "ukjent")
        chunk_nr = metadataer[i].get("chunk_nr", 0)
        avst = avstander[i]
        dok_vektor_tall = [round(float(v), 4) for v in dok_vektorer[i][:8]]
        print(f"     [{i}] Kilde: {kilde} (Chunk {chunk_nr}) | Avstand: {avst:.4f}")
        print(f"         Vektortall: {dok_vektor_tall} ...")
        
        utdrag_tekst = f"Fil: {kilde} (Del {chunk_nr})\nInnhold:\n{dokumenter[i]}"
        if utdrag_tekst not in filtrerte_kilder and len(filtrerte_kilder) < 5:
            filtrerte_kilder.append(utdrag_tekst)

    generer_og_lagre_visuell_figur(bruker_sporsmal, dokumenter, metadataer, avstander, sporsmal_vektor, dok_vektorer)

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    meldinger = [
        {
        "role": "system",
        "content": (
            "Du er en konsis dokumentassistent. Du svarer utelukkende basert på de oppgitte kildene, "
            "med naturlig, flytende norsk språk.\n\n"

            "REGLER FOR SVAR:\n"
            "1. Svar direkte på det som blir spurt om, uten å kommentere spørsmålsstillingen, "
            "brukerens antakelser eller hvorfor de spør.\n"
            "2. Hvis brukeren oppgir en verdi eller påstand som ikke stemmer med kildene, skal du "
            "korrigere dette naturlig i svaret – ikke bare bekrefte eller avkrefte, men gi riktig "
            "informasjon i samme setning. "
            "3. Hvis kildene ikke inneholder svaret, si tydelig at informasjonen ikke finnes i kildene "
            "du har tilgjengelig. Ikke gjett eller fyll inn med generell kunnskap.\n"
            "4. Ikke bruk faste innledningsfraser, maler eller unødvendige høflighetsfraser "
            "(f.eks. 'Takk for spørsmålet', 'Basert på kildene kan jeg fortelle at...'). "
            "Gå rett på svaret.\n"
            "5. Hold svarene korte og presise – ikke lengre enn nødvendig for å svare fullstendig.\n"
            "6. Hvis spørsmålet er tvetydig og kildene inneholder flere mulige svar, "
            "spesifiser hvilket du svarer ut ifra, uten å gjette på hva brukeren mente."
        )
    },
        {
            "role": "user", 
            "content": f"Spørsmål: {bruker_sporsmal}\n\nRelevante kilder:\n{kombinert_kontekst}"
        }
    ]

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)

    print("4. LLM-Generering: Sender prompt med kontekst til Qwen-modellen...")
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