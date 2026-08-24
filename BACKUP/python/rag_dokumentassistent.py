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

# ---------------------------------------------------------
# Stier til mapper
# ---------------------------------------------------------
MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"

# ---------------------------------------------------------
# 1. Oppsett av Chroma-database
# ---------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=DB_STI)

# Flerspråklig embedding-modell som forstår norsk
embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# VIKTIG: navnet her må matche navnet på samlingen som opprettes under,
# ellers blir gamle/duplikate biter liggende igjen i databasen permanent.
COLLECTION_NAVN = "mine_dokumenter"

try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass  # samlingen finnes ikke fra før - helt normalt ved første kjøring

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)


# ---------------------------------------------------------
# Deler tekst inn i chunks med overlapping basert på ord
# ---------------------------------------------------------
def del_tekst_i_chunks(tekst, chunk_storrelse=100, overlapping=20):
    ord_liste = tekst.split()
    chunks = []
    i = 0
    while i < len(ord_liste):
        chunk_ord = ord_liste[i: i + chunk_storrelse]
        chunk_tekst = " ".join(chunk_ord)
        chunks.append(chunk_tekst)
        if i + chunk_storrelse >= len(ord_liste):
            break
        i += (chunk_storrelse - overlapping)
    return chunks


# ---------------------------------------------------------
# Oppdaterer én fil i databasen
# ---------------------------------------------------------
def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"):
        return

    filnavn = os.path.basename(full_sti)

    # Fjern gamle biter fra denne filen først, uansett om filen fortsatt finnes
    try:
        db_samling.delete(where={"kilde": filnavn})
    except Exception:
        pass

    if not os.path.exists(full_sti):
        print(f"[Database] Fjernet {filnavn} (filen ble slettet).")
        return

    with open(full_sti, "r", encoding="utf-8") as fil:
        tekst = fil.read().strip()

    if not tekst:
        return

    # Mer overlapping (50%) reduserer risikoen for at én setning blir delt
    # over to chunks uten å være komplett i noen av dem
    chunks = del_tekst_i_chunks(tekst, chunk_storrelse=60, overlapping=30)

    dokumenter = []
    metadatas = []
    ids = []

    for indeks, chunk in enumerate(chunks):
        dokumenter.append(chunk)
        metadatas.append({"kilde": filnavn, "chunk_nr": indeks})
        ids.append(f"{filnavn}_chunk_{indeks}")

    db_samling.add(
        documents=dokumenter,
        metadatas=metadatas,
        ids=ids
    )
    print(f"[Database] Suksess! Delte {filnavn} inn i {len(chunks)} biter med overlapping.")


# ---------------------------------------------------------
# 2. Watchdog - overvåker mappen for endringer
# ---------------------------------------------------------
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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
bakgrunns_lytter.start()

# Gi watchdog litt tid til å fullføre førstegangs-indeksering før spørring starter
time.sleep(2)

# ---------------------------------------------------------
# 3. Qwen (generator-modellen)
# ---------------------------------------------------------
print("[System] Laster inn Qwen-modellen...")
ai_modell = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------
# 4. Spørsmåls-løkke
# ---------------------------------------------------------
while True:
    bruker_sporsmal = input("\nHva vil du spørre om? (eller skriv 'avslutt'): ")

    if bruker_sporsmal.lower().strip() == "avslutt":
        print("Avslutter programmet. Ha en fin dag!")
        break

    if not bruker_sporsmal.strip():
        continue

    print("Søker etter mest relevante tekst-biter...")

    # Hent flere kandidater - se forklaring under
    sok_resultat = db_samling.query(
        query_texts=[bruker_sporsmal],
        n_results=6
    )

    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        print("-> Svar fra AI: Fant ingen relevante dokumenter.")
        continue

    dokument_kontekster = sok_resultat["documents"][0]
    metadata_liste = sok_resultat["metadatas"][0]
    avstander = sok_resultat.get("distances", [[None] * len(dokument_kontekster)])[0]

    # --- Enkel hybrid-boost: nøkkelord fra spørsmålet teller i tillegg til vektor-avstand ---
    sporsmal_ord = set(w.lower().strip(".,?!") for w in bruker_sporsmal.split() if len(w) > 3)

    def nokkelord_score(tekst):
        tekst_ord = set(w.lower().strip(".,?!") for w in tekst.split())
        return len(sporsmal_ord & tekst_ord)

    kandidater = []
    for tekst_bit, meta, avstand in zip(dokument_kontekster, metadata_liste, avstander):
        kw_score = nokkelord_score(tekst_bit)
        kandidater.append((tekst_bit, meta, avstand, kw_score))

    # Sorter: flest nøkkelord-treff øverst, deretter lavest vektor-avstand
    kandidater.sort(key=lambda x: (-x[3], x[2]))

    # Debug-utskrift: vis hva som faktisk ble hentet ut, avstand og nøkkelord-treff
    print("Debug - hentede biter (sortert etter nøkkelord + avstand):")
    for idx, (tekst_bit, meta, avstand, kw_score) in enumerate(kandidater):
        forhandsvisning = tekst_bit[:80].replace("\n", " ")
        print(f"  [{idx}] kilde={meta['kilde']} chunk={meta['chunk_nr']} avstand={avstand:.4f} nokkelord={kw_score} | \"{forhandsvisning}...\"")

    # Bruk kun de 3 best rangerte etter hybrid-sortering til selve konteksten
    beste_kandidater = kandidater[:3]
    dokument_kontekster = [k[0] for k in beste_kandidater]
    metadata_liste = [k[1] for k in beste_kandidater]

    kombinert_kontekst = "\n---\n".join(dokument_kontekster)
    kilde_fil = metadata_liste[0]["kilde"]
    chunk_nr = metadata_liste[0]["chunk_nr"]

    print(f"AI-en analyserer {len(dokument_kontekster)} bit(er), best treff fra: {kilde_fil}...")

    meldinger = [
        {
            "role": "system",
            "content": (
                "Du er en presis assistent. Svar kort med én setning på norsk, "
                "basert KUN på teksten under. Bruk den biten som faktisk svarer "
                "på spørsmålet, ikke nødvendigvis den første.\n\n"
                f"{kombinert_kontekst}\n\n"
                "Hvis ingen av bitene inneholder svaret, svar nøyaktig: "
                "'Jeg finner ikke svar i dokumentet.'"
            ),
        },
        {"role": "user", "content": bruker_sporsmal},
    ]

    prompt = ai_modell.tokenizer.apply_chat_template(
        meldinger, tokenize=False, add_generation_prompt=True
    )

    resultat = ai_modell(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    full_generert_tekst = resultat[0]["generated_text"]
    svar_linjer = full_generert_tekst.split("\n")
    svar = svar_linjer[0].strip() if svar_linjer else ""

    if svar and not svar.endswith("."):
        svar += "."

    print("\n")
    print(f"-> Svar fra AI: [{kilde_fil} | Chunk {chunk_nr}] {svar}")
    print("\n")
