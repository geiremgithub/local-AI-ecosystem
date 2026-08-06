import os
import sys
import warnings
import time
from threading import Thread, Event

# Sørg for at terminalen skriver ut norske tegn (æ, ø, å) riktig,
# selv om Windows-konsollet i utgangspunktet bruker en gammel kodeside
# som ikke støtter UTF-8. Uten denne linjen kan man få rare tegn som
# "�" i stedet for "ø" i utskriften.
sys.stdout.reconfigure(encoding="utf-8")

# Skru av unødvendige advarsler i terminalen
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging

# Demp transformers sitt EGET interne loggesystem (adskilt fra Pythons
# vanlige "warnings"-modul, se forklaring lenger nede i koden ved
# generation_config). Informasjonsmeldinger som "Both max_new_tokens..."
# og "generation flags are not valid..." kommer herfra, og lar seg ikke
# fjerne med warnings.filterwarnings() alene. set_verbosity_error() gjør
# at kun ekte feilmeldinger (errors) vises, ikke rene informasjons- og
# advarselsmeldinger om intern konfigurasjon.
hf_logging.set_verbosity_error()
import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# Deler tekst inn i chunks med overlapping - nå med
# RecursiveCharacterTextSplitter, som splitter på avsnitt/
# linjeskift/setninger FØR den i det hele tatt vurderer å
# kutte midt i en setning. Dette er langt tryggere for
# nummererte punkter og korte faktasetninger enn ren
# ord-telling.
# ---------------------------------------------------------
tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,       # antall tegn per chunk (ca. 50-60 ord på norsk)
    chunk_overlap=100,    # overlapping i tegn mellom chunks
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def del_tekst_i_chunks(tekst, chunk_storrelse=None, overlapping=None):
    # chunk_storrelse/overlapping-parameterne beholdes for bakoverkompatibilitet
    # med resten av koden, men selve splittingen styres nå av tekst_splitter
    # (juster chunk_size/chunk_overlap over hvis du vil endre oppførselen).
    return tekst_splitter.split_text(tekst)


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

    # Les filen robust: prøv UTF-8 først (standard og anbefalt), men gi
    # ikke opp om filen er lagret i et annet tegnsett (f.eks. "ANSI"/
    # Windows-1252, vanlig hvis filen er laget i Notisblokk på norsk
    # Windows). Uten dette ville ÉN feilkodet fil krasje HELE
    # bakgrunnstråden og stoppe programmet fra å noensinne bli klart
    # for spørsmål.
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


# Signaliserer til hovedtråden når førstegangs-indekseringen faktisk er
# ferdig, i stedet for å gjette med en fast time.sleep()-pause. Uten dette
# kan hovedtråden rekke å starte spørsmåls-løkken FØR indekseringen er
# ferdig, slik at print()-linjer fra de to trådene blander seg sammen på
# skjermen og det ser ut som om programmet "henger".
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
        # Uansett hva som går galt under oppstart av overvåkingen, skal
        # IKKE hovedtråden bli hengende for evig og vente på et signal
        # som aldri kommer. Vi skriver ut feilen tydelig i stedet.
        print(f"[Feil] Noe gikk galt under oppstart av mappeovervåking: {feil}")
        forste_indeksering_ferdig.set()
        return

    # Nå er all førstegangsindeksering faktisk ferdig - gi beskjed til
    # hovedtråden om at det er trygt å starte spørsmåls-løkken.
    forste_indeksering_ferdig.set()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
bakgrunns_lytter.start()

# Vent til bakgrunnstråden faktisk har fullført indekseringen av alle
# filene (i stedet for å gjette med en fast pause). Timeout=60 er et
# sikkerhetsnett: skulle noe helt uforutsett likevel få tråden til å
# henge (i stedet for å feile ryddig), gir vi opp å vente etter 60
# sekunder og starter spørsmåls-løkken uansett, med en tydelig advarsel,
# fremfor å blokkere programmet for alltid.
if not forste_indeksering_ferdig.wait(timeout=60):
    print("[Advarsel] Indekseringen brukte uventet lang tid (>60s). "
          "Starter spørsmåls-løkken uansett - databasen kan være ufullstendig.")

# ---------------------------------------------------------
# 3. Qwen (generator-modellen)
# ---------------------------------------------------------
print("[System] Laster inn Qwen-modellen...")
ai_modell = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------
# Lager EN eksplisitt, isolert GenerationConfig i stedet for å stole på
# (og prøve å rette opp i) modellens skjulte standardkonfigurasjon.
#
# Dette er den offisielt anbefalte fremgangsmåten fra transformers selv:
# advarselen biblioteket ellers gir sier bokstavelig talt "pass enten et
# generation_config-objekt ELLER alle parametere eksplisitt, men ikke
# begge deler". Ved å bygge vårt eget objekt her unngår vi at gamle
# standardverdier fra Qwen sin medfølgende konfigurasjon (som
# max_length=20) kan lekke inn og kollidere med det vi faktisk vil bruke.
#
# Merk: temperature/top_p/top_k er IKKE satt her i det hele tatt - de er
# kun relevante når do_sample=True og ignoreres automatisk når
# do_sample=False. Rene informasjonsmeldinger om dette er dempet ved
# kilden med hf_logging.set_verbosity_error() øverst i skriptet.
# ---------------------------------------------------------
generasjons_konfig = GenerationConfig(
    max_new_tokens=50,   # maks antall NYE tokens i selve svaret
    max_length=None,     # nullstill den gamle "total lengde"-grensen
    do_sample=False,     # deterministisk - velg alltid mest sannsynlige token
)

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

    # Sorter: flest nøkkelord-treff øverst, deretter lavest vektor-avstand.
    # (avstand kan i teorien være None hvis Chroma ikke returnerer distances -
    # da sorterer vi bare på nøkkelord-score for å unngå krasj.)
    kandidater.sort(key=lambda x: (-x[3], x[2] if x[2] is not None else float("inf")))

    # Debug-utskrift: vis hva som faktisk ble hentet ut, avstand og nøkkelord-treff
    print("Debug - hentede biter (sortert etter nøkkelord + avstand):")
    for idx, (tekst_bit, meta, avstand, kw_score) in enumerate(kandidater):
        forhandsvisning = tekst_bit[:80].replace("\n", " ")
        avstand_tekst = f"{avstand:.4f}" if avstand is not None else "N/A"
        print(f"  [{idx}] kilde={meta['kilde']} chunk={meta['chunk_nr']} avstand={avstand_tekst} nokkelord={kw_score} | \"{forhandsvisning}...\"")

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
        generation_config=generasjons_konfig,
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
