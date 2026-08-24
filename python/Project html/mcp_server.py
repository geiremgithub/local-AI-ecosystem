import os
import sys
import time
import warnings
from threading import Thread, Event

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mcp.server.fastmcp import FastMCP

SKRIPT_VERSJON = (
    "v1-mcp - gjenbruker mappeovervåkingen og Chroma-indekseringen fra "
    "rag_server.py, men dropper Qwen-generatoren. Eksponerer rådokumentsøk "
    "som MCP-verktøy, slik at en MCP-klient (f.eks. Claude Desktop) kan "
    "søke i de samme dokumentene og selv formulere svaret."
)

# -----------------------------------------------------------------
# VIKTIG om logging i en MCP-server (stdio-transport):
# All kommunikasjon mellom serveren og MCP-klienten går over stdout.
# Et helt vanlig print() dit vil ødelegge protokollen og få klienten
# til å miste kontakten med serveren. Derfor skrives ALL logging i
# denne fila til stderr - i motsetning til rag_server.py, som trygt
# kan bruke vanlig print() siden den ikke snakker MCP.
# -----------------------------------------------------------------
def logg(tekst):
    print(tekst, file=sys.stderr, flush=True)


logg(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

# -----------------------------------------------------------------
# Stier - MÅ peke på samme dokumentmappe som rag_server.py for å
# søke i de samme filene.
#
# ChromaDB-stien (DB_STI) kan enten:
#   a) være den SAMME som i rag_server.py, MEN da må du aldri kjøre
#      rag_server.py og denne MCP-serveren samtidig - begge
#      overvåker mappen og skriver til databasen uavhengig av
#      hverandre, og rag_server.py sletter i tillegg hele samlingen
#      ved hver oppstart, noe som vil rive vekk det MCP-serveren
#      nettopp har indeksert (og omvendt).
#   b) peke på en EGEN mappe (f.eks. DB_STI + "_mcp"), slik at de to
#      serverne har hver sin uavhengige database og trygt kan kjøre
#      samtidig. Det koster litt ekstra diskplass og dobbel
#      embedding-tid ved oppstart, men er tryggest.
#
# Under er (a) valgt for enkelhets skyld - bytt til (b) hvis du vil
# kunne kjøre nettsiden og MCP-serveren samtidig.
# -----------------------------------------------------------------
MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"

chroma_client = chromadb.PersistentClient(path=DB_STI)

embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAVN = "mine_dokumenter"

# Merk: i motsetning til rag_server.py sletter vi IKKE samlingen ved
# oppstart her - vi henter den (eller oppretter den om den ikke
# finnes) og lar per-fil-oppdateringen under holde den i sync. Det
# gjør MCP-serveren trygg å starte på nytt uten å måtte reindeksere
# alt fra bunnen hver gang.
db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell,
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
        logg(f"[Database] Fjernet {filnavn} (filen ble slettet).")
        return

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
    except UnicodeDecodeError:
        try:
            with open(full_sti, "r", encoding="cp1252") as fil:
                tekst = fil.read().strip()
        except Exception as feil:
            logg(f"[Feil] Klarte ikke å lese {filnavn} uansett tegnsett: {feil}")
            return
    except Exception as feil:
        logg(f"[Feil] Uventet feil ved lesing av {filnavn}: {feil}")
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
    logg(f"[Database] Indekserte {filnavn} i {len(chunks)} biter.")


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
        if not os.path.exists(MAPPESTI):
            os.makedirs(MAPPESTI)

        for filnavn in os.listdir(MAPPESTI):
            oppdater_fil_i_database(os.path.join(MAPPESTI, filnavn))

        handler = DokumentLytter()
        observer = Observer()
        observer.schedule(handler, path=MAPPESTI, recursive=False)
        observer.start()
        logg(f"[Watchdog] Lytter aktivt i: {MAPPESTI}")
    except Exception as feil:
        logg(f"[Feil] Noe gikk galt under oppstart av mappeovervåking: {feil}")
        forste_indeksering_ferdig.set()
        return

    forste_indeksering_ferdig.set()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# -----------------------------------------------------------------
# Delt søkelogikk - samme hybrid-rangering (vektorsøk + enkelt
# nøkkelord-score som tiebreaker) som besvar_sporsmal() bruker i
# rag_server.py, men stopper her: ingen Qwen-kall. MCP-klienten
# (f.eks. Claude) gjør selve resonneringen og formuleringen av
# svaret ut fra tekstbitene dette verktøyet returnerer.
# -----------------------------------------------------------------
def sok_med_reranking(sporsmal, antall_treff):
    sok_resultat = db_samling.query(
        query_texts=[sporsmal], n_results=max(antall_treff * 2, 8)
    )

    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        return []

    dokument_kontekster = sok_resultat["documents"][0]
    metadata_liste = sok_resultat["metadatas"][0]
    avstander = sok_resultat.get("distances", [[None] * len(dokument_kontekster)])[0]

    sporsmal_ord = set(w.lower().strip(".,?!") for w in sporsmal.split() if len(w) > 3)

    def nokkelord_score(tekst):
        tekst_ord = set(w.lower().strip(".,?!") for w in tekst.split())
        return len(sporsmal_ord & tekst_ord)

    kandidater = []
    for tekst_bit, meta, avstand in zip(dokument_kontekster, metadata_liste, avstander):
        kandidater.append((tekst_bit, meta, avstand, nokkelord_score(tekst_bit)))

    kandidater.sort(key=lambda x: (-x[3], x[2] if x[2] is not None else float("inf")))

    return kandidater[:antall_treff]


# -----------------------------------------------------------------
# MCP-serveren og verktøyene den eksponerer
# -----------------------------------------------------------------
mcp = FastMCP("dokumentoppslag")


@mcp.tool()
def sok_i_dokumenter(sporsmal: str, antall_treff: int = 5) -> str:
    """
    Søk i den lokale dokumentmappen (samme dokumenter som brukes i
    Dokumentoppslag-nettsiden) og hent ut de mest relevante
    tekstbitene for et spørsmål eller en påstand.

    Bruk dette verktøyet FØR du svarer på spørsmål om innholdet i
    dokumentene - ikke gjett eller bruk generell kunnskap. Svaret
    inneholder tekstbitene med kildefil og bit-nummer, slik at du kan
    oppgi nøyaktig hvor informasjonen kommer fra. Hvis ingen treff er
    relevante for spørsmålet, si tydelig fra at dokumentene ikke
    later til å inneholde svaret - ikke finn på noe.

    Args:
        sporsmal: Spørsmålet eller påstanden som skal slås opp.
        antall_treff: Hvor mange tekstbiter som skal returneres (standard 5).
    """
    treff = sok_med_reranking(sporsmal, antall_treff)

    if not treff:
        return "Ingen relevante dokumenter ble funnet for dette spørsmålet."

    deler = []
    for i, (tekst, meta, _avstand, _score) in enumerate(treff, start=1):
        deler.append(
            f"TREFF {i} (kilde: {meta['kilde']}, bit {meta['chunk_nr']}):\n{tekst}"
        )
    return "\n\n".join(deler)


@mcp.tool()
def list_dokumenter() -> str:
    """
    List opp alle dokumentfiler som for øyeblikket er indeksert og
    søkbare. Bruk dette for å se hvilke kilder som finnes før du
    søker, eller for å svare på spørsmål om hvilke dokumenter som
    er tilgjengelige i mappen.
    """
    alle = db_samling.get(include=["metadatas"])
    kilder = sorted(set(m["kilde"] for m in alle["metadatas"]))
    if not kilder:
        return "Ingen dokumenter er indeksert ennå."
    return "\n".join(kilder)


@mcp.tool()
def hent_hele_dokumentet(filnavn: str) -> str:
    """
    Hent hele det opprinnelige innholdet til én dokumentfil direkte
    fra disk, ikke bare enkeltbiter. Bruk dette når treffene fra
    sok_i_dokumenter ikke gir nok sammenheng og du trenger å lese
    hele dokumentet i sin helhet.

    Args:
        filnavn: Filnavnet slik det vises i sok_i_dokumenter/list_dokumenter,
            f.eks. "personalhandbok.txt".
    """
    full_sti = os.path.join(MAPPESTI, filnavn)

    # Enkel beskyttelse mot at filnavn med "../" leder utenfor dokumentmappen.
    if not os.path.abspath(full_sti).startswith(os.path.abspath(MAPPESTI)):
        return "Ugyldig filnavn."

    if not os.path.exists(full_sti):
        return f"Fant ikke filen {filnavn} i dokumentmappen."

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            return fil.read()
    except UnicodeDecodeError:
        with open(full_sti, "r", encoding="cp1252") as fil:
            return fil.read()


if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    if not forste_indeksering_ferdig.wait(timeout=60):
        logg(
            "[Advarsel] Indekseringen brukte uventet lang tid (>60s). "
            "Starter MCP-serveren uansett - databasen kan være ufullstendig."
        )

    logg("[System] MCP-serveren er klar og lytter på stdio.")
    mcp.run(transport="stdio")
