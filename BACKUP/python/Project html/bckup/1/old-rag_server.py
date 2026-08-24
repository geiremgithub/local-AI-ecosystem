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

SKRIPT_VERSJON = "v17-web - samme motor som v16, men med Flask-API i stedet for terminal-løkke"
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
# 1. Oppsett av Chroma-database (uendret fra terminal-versjonen)
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
# 3. Qwen (generator-modellen) - lastes én gang når serveren starter,
# ikke på nytt for hvert spørsmål (det ville vært svært tregt).
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
        "uttrykk. Eksempel på tonen: i stedet for 'Skolen starter klokken "
        "08:30', skriv noe sånt som 'Sett vekkerklokken - skolen smeller i "
        "gang presis 08:30!'. ALDRI på bekostning av at all "
        "faktainformasjon fra FAKTA fortsatt er korrekt med."
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

# ---------------------------------------------------------
# Enkel samtalehistorikk (kun ett globalt minne - denne appen kjører
# lokalt for én bruker om gangen). Brukes til å gi søket og modellen
# kontekst fra forrige runde, slik at oppfølgingsspørsmål som
# "det gjelder snus også" forstås riktig.
# ---------------------------------------------------------
SAMTALE_HISTORIKK = []  # liste av {"sporsmal": ..., "svar": ...}
MAKS_HISTORIKK = 3


def bygg_sok_sporsmal(bruker_sporsmal):
    """Lager en kontekst-beriket versjon av spørsmålet til bruk i vektorsøket,
    slik at korte oppfølgingsspørsmål ("gjelder det snus også?") arver
    temaet fra forrige spørsmål i stedet for å søkes helt isolert."""
    if not SAMTALE_HISTORIKK:
        return bruker_sporsmal
    forrige = SAMTALE_HISTORIKK[-1]["sporsmal"]
    return f"{forrige} {bruker_sporsmal}"


# ---------------------------------------------------------
# 4. Selve spørre-funksjonen - dette ER innholdet i den gamle
# while-løkken din, bare omgjort til en funksjon som returnerer et
# dict i stedet for å printe til terminalen. Flask-endepunktet under
# kaller denne for hvert spørsmål som kommer fra nettleseren.
# ---------------------------------------------------------
def besvar_sporsmal(bruker_sporsmal, gjeldende_modus):
    if gjeldende_modus not in MODUS_INSTRUKSER:
        gjeldende_modus = STANDARD_MODUS

    sok_sporsmal = bygg_sok_sporsmal(bruker_sporsmal)
    sok_resultat = db_samling.query(query_texts=[sok_sporsmal], n_results=8)

    if not sok_resultat["documents"] or not sok_resultat["documents"][0]:
        return {"svar": "Fant ingen relevante dokumenter.", "fakta": "", "sammenligning": "",
                "kilde": "", "chunk": None, "advarsel": None}

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

    beste_kandidater = kandidater[:5]
    dokument_kontekster = [k[0] for k in beste_kandidater]
    metadata_liste = [k[1] for k in beste_kandidater]

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
                "SAMMENLIGNING: <kun hvis brukeren skrev en PÅSTAND: skriv "
                "nøyaktig hva brukeren hevdet OG hva FAKTA faktisk sier, side "
                "om side. Hvis brukeren stilte et RENT SPØRSMÅL (spurte om noe, "
                "hevdet ikke noe selv), skriv 'ikke aktuelt - rent spørsmål'>\n"
                "SVAR: <kort svar på norsk, én setning>\n\n"
                "VIKTIG: ikke dikt opp en påstand hvis brukeren bare stilte et "
                "spørsmål. Hvis brukerens påstand og FAKTA inneholder ULIKE "
                "tall/tider/verdier, MÅ SVAR starte med 'Nei, det stemmer ikke'. "
                "Hvis de er IDENTISKE, skal SVAR starte med 'Ja, det stemmer'. "
                "Hvis det var et rent spørsmål, svar direkte uten "
                "'Ja/Nei, det stemmer'.\n\n"
                "VIKTIG: SVAR skal inneholde ALL relevant informasjon fra "
                "FAKTA-linjen - ikke forkort eller utelat detaljer (unntak, "
                "tidsangivelser, betingelser osv.) når du omformulerer til SVAR.\n\n"
                "VIKTIG - ALDRI MOTSI DIN EGEN FAKTA-LINJE: hvis FAKTA "
                "inneholder faktisk informasjon (altså IKKE er '(ingen)'), "
                "MÅ SVAR bygge på og gjengi akkurat den informasjonen. SVAR "
                "skal ALDRI si noe sånt som 'jeg finner ingen informasjon' "
                "eller 'ingen relevant informasjon' når FAKTA-linjen din "
                "selv inneholder relevant informasjon - det er en "
                "selvmotsigelse. 'Jeg finner ikke svar i dokumentet' skal "
                "KUN brukes når FAKTA faktisk er '(ingen)'.\n\n"
                "VIKTIG - TERSKEL-/VILKÅRSSPØRSMÅL: hvis spørsmålet lurer på om "
                "en PERSON, ALDER, DATO eller VERDI oppfyller et krav som er "
                "nevnt i FAKTA (f.eks. spørsmål av typen 'kan en X-åring...', "
                "'gjelder dette for...', 'har person Y rett til...'), er dette "
                "ALLTID en påstand som skal sjekkes, selv om det er formulert "
                "som et spørsmål. IKKE svar 'Ja' bare fordi nøkkelord (som "
                "f.eks. selve navnet på ordningen) finnes i FAKTA - du MÅ "
                "regne ut om verdien i spørsmålet faktisk oppfyller kravet "
                "(er den STØRRE ELLER LIK, eller MINDRE ENN grensen i FAKTA?) "
                "og skrive dette eksplisitt i SAMMENLIGNING før du konkluderer "
                "i SVAR.\n\n"
                "VIKTIG - JA/NEI TILLATELSESSPØRSMÅL (f.eks. 'er det lov å...', "
                "'kan man...', 'har man rett til...'): dette er ALLTID et "
                "spørsmål som skal sjekkes opp mot FAKTA, på samme måte som "
                "terskel-/vilkårsspørsmål over. Se etter om FAKTA beskriver "
                "noe som FORBUDT/IKKE tillatt eller som TILLATT/lov, og sørg "
                "for at polariteten i SVAR stemmer nøyaktig med FAKTA - skriv "
                "dette eksplisitt i SAMMENLIGNING før du konkluderer. Hvis "
                "FAKTA sier noe er forbudt/ikke tillatt, MÅ SVAR si klart at "
                "det IKKE er lov (start med 'Nei, det er ikke lov...'). Hvis "
                "FAKTA sier noe er tillatt, MÅ SVAR si klart at det ER lov "
                "(start med 'Ja, det er lov...'). SVAR skal ALDRI si 'lov', "
                "'tillatt' eller lignende bekreftende ord når FAKTA beskriver "
                "et forbud - det er en selvmotsigelse akkurat som å si "
                "'ingen informasjon' når FAKTA fant noe.\n\n"
                "VIKTIG - INGEN OPPDIKTEDE ORD: SVAR skal KUN bruke ekte, "
                "korrekt stavede norske ord. Ikke lag sammensatte eller "
                "forvrengte ord som ikke finnes i det norske språket (f.eks. "
                "'lovligferdigt' er IKKE et ord). Bruk vanlige, riktige "
                "norske ord som 'lovlig', 'ulovlig', 'tillatt' og 'forbudt'.\n\n"
                "VIKTIG: eksemplene under (Eksempel 1-5) viser KUN hvordan "
                "SVAR skal formuleres og struktures. Tallene og påstandene i "
                "dem er oppdiktet illustrasjon - de har INGENTING med de "
                "ekte dokumentene å gjøre, selv om et tall i eksemplene "
                "skulle ligne et tall i brukerens spørsmål. FAKTA skal "
                "ALLTID hentes ordrett fra TEKSTBIT-ene lenger ned (etter "
                "skillelinjen '----- EKTE DOKUMENTINNHOLD -----'), ALDRI fra "
                "eksemplene.\n\n"
                "Eksempel 1 (påstand):\n"
                "Bruker: 'byggesøknaden ble sendt inn 3. mars'\n"
                "FAKTA: Byggesøknaden ble registrert mottatt 17. juni.\n"
                "SAMMENLIGNING: Bruker hevdet 3. mars. FAKTA sier 17. juni. "
                "3. mars er IKKE det samme som 17. juni.\n"
                "SVAR: Nei, det stemmer ikke - byggesøknaden ble registrert "
                "mottatt 17. juni, ikke 3. mars.\n\n"
                "Eksempel 2 (rent spørsmål, med flere detaljer som MÅ bevares):\n"
                "Bruker: 'når er eksamen'\n"
                "FAKTA: Eksamen arrangeres utelukkende på hverdager og aldri "
                "i helger eller på helligdager.\n"
                "SAMMENLIGNING: ikke aktuelt - rent spørsmål\n"
                "SVAR: Eksamen arrangeres utelukkende på hverdager, og aldri "
                "i helger eller på helligdager.\n\n"
                "Eksempel 3 (terskel-/vilkårsspørsmål - verdi under grensen):\n"
                "Bruker: 'kan en ansatt med 2 års ansiennitet få gullklokke?'\n"
                "FAKTA: Gullklokke tildeles ansatte med minst 25 års "
                "ansiennitet i bedriften.\n"
                "SAMMENLIGNING: Spørsmålet gjelder 2 års ansiennitet. FAKTA "
                "krever minst 25 år. 2 er MINDRE ENN 25, altså er kravet "
                "IKKE oppfylt.\n"
                "SVAR: Nei, en ansatt med 2 års ansiennitet får ikke "
                "gullklokke ennå - kravet er minst 25 års ansiennitet.\n\n"
                "Eksempel 4 (terskel-/vilkårsspørsmål - verdi over/på grensen):\n"
                "Bruker: 'kan en ansatt med 30 års ansiennitet få gullklokke?'\n"
                "FAKTA: Gullklokke tildeles ansatte med minst 25 års "
                "ansiennitet i bedriften.\n"
                "SAMMENLIGNING: Spørsmålet gjelder 30 års ansiennitet. FAKTA "
                "krever minst 25 år. 30 er STØRRE ENN 25, altså er kravet "
                "oppfylt.\n"
                "SVAR: Ja, en ansatt med 30 års ansiennitet får gullklokke, "
                "siden kravet om minst 25 års ansiennitet er oppfylt.\n\n"
                "Eksempel 5 (ja/nei tillatelsesspørsmål - noe er forbudt):\n"
                "Bruker: 'er det lov å røyke på skolen?'\n"
                "FAKTA: Skolens område er helt røykfritt, og dette forbudet "
                "inkluderer også all bruk av e-sigaretter og snus.\n"
                "SAMMENLIGNING: Spørsmålet gjelder om røyking er lov. FAKTA "
                "sier området er røykfritt, altså er røyking FORBUDT, ikke "
                "lov.\n"
                "SVAR: Nei, det er ikke lov å røyke på skolen - området er "
                "helt røykfritt, og forbudet inkluderer også e-sigaretter "
                "og snus.\n\n"
                "----- EKTE DOKUMENTINNHOLD (dette og KUN dette skal FAKTA "
                "hentes fra) -----\n"
                f"{kombinert_kontekst}\n\n"
                "Hvis INGEN tekstbit over er relevant, skriv:\n"
                "FAKTA: (ingen)\n"
                "SAMMENLIGNING: ikke aktuelt\n"
                "SVAR: Jeg finner ikke svar i dokumentet.\n\n"
                f"TONE FOR SVAR-LINJEN: {MODUS_INSTRUKSER[gjeldende_modus]}"
            ),
        },
    ]

    # Legg inn tidligere spørsmål/svar som ekte samtaleturer, slik at modellen
    # forstår hva et kort oppfølgingsspørsmål faktisk refererer til.
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

    # Sikkerhetsnett #0: sjekk at FAKTA-linjen faktisk stammer fra de ekte
    # tekstbitene, og ikke fra et av few-shot-eksemplene i system-prompten
    # (dette er akkurat det som skjedde med "gullklokke"-hallusinasjonen:
    # modellen kopierte Eksempel 4 ordrett i stedet for å bruke ekte data).
    def fakta_finnes_i_kontekst(fakta, kontekst_biter):
        if not fakta or fakta.strip().lower() in ("(ingen)", "ingen", ""):
            return True  # "ingen funnet" er alltid gyldig, sjekkes ikke her
        fakta_ord = set(w.lower().strip(".,?!") for w in fakta.split() if len(w) > 3)
        if not fakta_ord:
            return True
        beste_overlapp = 0.0
        for bit in kontekst_biter:
            bit_ord = set(w.lower().strip(".,?!") for w in bit.split() if len(w) > 3)
            if not bit_ord:
                continue
            overlapp = len(fakta_ord & bit_ord) / len(fakta_ord)
            beste_overlapp = max(beste_overlapp, overlapp)
        return beste_overlapp >= 0.6  # minst 60% av FAKTA-ordene må faktisk finnes igjen

    def finn_kilde_for_fakta(fakta, kontekst_biter, metadata):
        fakta_ord = set(w.lower().strip(".,?!") for w in fakta.split() if len(w) > 3)
        beste_indeks, beste_overlapp = 0, -1
        for i, bit in enumerate(kontekst_biter):
            bit_ord = set(w.lower().strip(".,?!") for w in bit.split())
            overlapp = len(fakta_ord & bit_ord)
            if overlapp > beste_overlapp:
                beste_overlapp, beste_indeks = overlapp, i
        return metadata[beste_indeks]["kilde"], metadata[beste_indeks]["chunk_nr"]

    kilde_fil, chunk_nr = finn_kilde_for_fakta(fakta_linje, dokument_kontekster, metadata_liste)

    if not fakta_finnes_i_kontekst(fakta_linje, dokument_kontekster):
        return {
            "svar": "Fant ikke noe pålitelig svar i dokumentene på dette spørsmålet.",
            "fakta": fakta_linje,
            "sammenligning": sammenligning_linje,
            "kilde": "",
            "chunk": None,
            "advarsel": (
                "Modellens FAKTA-linje samsvarte ikke med innholdet i de "
                "hentede tekstbitene (sannsynlig hallusinasjon, f.eks. fra "
                "et eksempel i prompten) - svaret er derfor blokkert i "
                "stedet for å vises."
            ),
        }

    # Samme kode-baserte sikkerhetsnett for alderskrav som i terminal-versjonen
    advarsel = None
    krav_match = re.search(r"fylte\s+(\d{1,3})\s*år", fakta_linje, re.IGNORECASE)
    sporsmal_alder_match = re.search(r"(\d{1,3})\s*[- ]?år", bruker_sporsmal, re.IGNORECASE)
    if krav_match and sporsmal_alder_match:
        krav_alder = int(krav_match.group(1))
        sporsmal_alder = int(sporsmal_alder_match.group(1))
        oppfylt = sporsmal_alder >= krav_alder
        svar_sier_ja = svar.strip().lower().startswith("ja")
        if oppfylt != svar_sier_ja:
            advarsel = (
                f"Mulig feil i svaret: FAKTA krever fylte {krav_alder} år, "
                f"spørsmålet gjelder {sporsmal_alder} år - vilkåret er "
                f"{'OPPFYLT' if oppfylt else 'IKKE oppfylt'}, men svaret "
                f"starter med {'Ja' if svar_sier_ja else 'Nei/annet'}."
            )

    # Sikkerhetsnett #2: modellen sier noen ganger "fant ingen informasjon" i
    # SVAR, selv om FAKTA-linjen (samme svar!) faktisk fant noe relevant.
    # Fanger opp denne selvmotsigelsen og retter SVAR automatisk i stedet for
    # å vise brukeren et feilaktig "ingen informasjon"-svar.
    fakta_fant_noe = bool(fakta_linje) and fakta_linje.strip().lower() not in ("(ingen)", "ingen", "")
    svar_sier_ingenting_funnet = any(
        frase in svar.lower()
        for frase in [
            "finner ikke svar", "finner ingen", "ingen relevant",
            "ikke funnet noe", "fant ingen", "ingen informasjon",
        ]
    )
    if fakta_fant_noe and svar_sier_ingenting_funnet:
        korrigert_svar = fakta_linje.strip()
        if not korrigert_svar.endswith("."):
            korrigert_svar += "."
        advarsel = (
            (advarsel + " " if advarsel else "") +
            "Det opprinnelige svaret motsa modellens egen FAKTA-linje "
            "(sa 'fant ingenting' selv om FAKTA fant noe relevant) - "
            "svaret under er derfor automatisk korrigert til å bruke "
            "FAKTA direkte."
        )
        svar = korrigert_svar

    SAMTALE_HISTORIKK.append({"sporsmal": bruker_sporsmal, "svar": svar})
    del SAMTALE_HISTORIKK[:-MAKS_HISTORIKK]

    return {
        "svar": svar,
        "fakta": fakta_linje,
        "sammenligning": sammenligning_linje,
        "kilde": kilde_fil,
        "chunk": chunk_nr,
        "advarsel": advarsel,
    }


# ---------------------------------------------------------
# 5. Flask-appen - dette er det som gjør spørsmålene tilgjengelige
# fra en nettleser i stedet for terminalen.
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
        print("[Advarsel] Indekseringen brukte uventet lang tid (>60s). "
              "Starter serveren uansett - databasen kan være ufullstendig.")

    print("[System] Serveren er klar. Åpne http://127.0.0.1:5000 i nettleseren din.")
    app.run(host="127.0.0.1", port=5000, debug=False)
