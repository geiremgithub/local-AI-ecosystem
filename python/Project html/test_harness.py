#!/usr/bin/env python3
"""
Test-harness for Q&A-testsett (skole-FAQ).

Bruk:
    python test_harness.py testsett.json

Filen testsett.json skal ha formatet:
[
  {"sporsmal": "...", "forventet": "..."},
  ...
]

Du plugger inn systemet ditt i funksjonen `sporr_systemet()` nederst.
"""

import json
import sys
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1) KOBLE TIL SYSTEMET DITT HER
# ---------------------------------------------------------------------------
import os
import requests

API_URL = os.environ.get("CHATBOT_API_URL", "http://127.0.0.1:5000/api/sporsmal")
MODUS = os.environ.get("CHATBOT_MODUS", "høflig")  # se MODUS_INSTRUKSER i scriptet ditt


def sporr_systemet(sporsmal: str) -> str:
    """
    Kaller Flask-endepunktet /api/sporsmal i skriptet ditt (v17-web).

    Request:  POST {"sporsmal": "...", "modus": "høflig"}
    Response: {"svar": "...", "fakta": "...", "sammenligning": "...",
               "kilde": "...", "chunk": ..., "advarsel": ... | None}
    """
    respons = requests.post(
        API_URL,
        json={"sporsmal": sporsmal, "modus": MODUS},
        timeout=120,  # modellen genererer tekst - kan ta tid, spesielt på CPU
    )
    respons.raise_for_status()
    data = respons.json()

    if "feil" in data:
        raise RuntimeError(f"API returnerte feil: {data['feil']}")

    return data  # returnerer HELE svaret (svar, fakta, kilde, advarsel, ...)


def nullstill_samtale():
    """Kaller /api/nullstill for å tømme SAMTALE_HISTORIKK før hver test,
    slik at ett spørsmåls svar ikke smitter over på det neste (uavhengige tester)."""
    nullstill_url = API_URL.rsplit("/", 1)[0] + "/nullstill"
    try:
        requests.post(nullstill_url, timeout=10)
    except requests.RequestException as e:
        print(f"    [advarsel] Klarte ikke å nullstille samtalehistorikk: {e}")


# ---------------------------------------------------------------------------
# 2) SAMMENLIGNINGSLOGIKK
# ---------------------------------------------------------------------------
def normaliser(tekst: str) -> str:
    """Gjør sammenligning mer robust: små bokstaver, trim, fjern punktum/mellomrom-støy."""
    tekst = tekst.strip().lower()
    tekst = re.sub(r"[.!?]+$", "", tekst)
    tekst = re.sub(r"\s+", " ", tekst)
    return tekst


def er_match(forventet: str, faktisk: str) -> bool:
    """
    Standard: eksakt match etter normalisering, ELLER at forventet
    svar er en delstreng av det faktiske svaret (nyttig hvis systemet
    svarer i fulle setninger, f.eks. "Skolen starter klokken 08:30").
    """
    f_norm = normaliser(forventet)
    a_norm = normaliser(faktisk)
    return f_norm == a_norm or f_norm in a_norm


# ---------------------------------------------------------------------------
# 3) HARNESS-LOGIKK
# ---------------------------------------------------------------------------
@dataclass
class TestResultat:
    sporsmal: str
    forventet: str
    faktisk: Optional[str]
    bestatt: bool
    feilmelding: Optional[str] = None
    advarsel: Optional[str] = None
    kilde: Optional[str] = None
    forventet_kilde: Optional[str] = None
    kilde_stemmer: Optional[bool] = None


def kjor_tester(testsett_path: str) -> list[TestResultat]:
    with open(testsett_path, "r", encoding="utf-8") as f:
        testsett = json.load(f)

    resultater = []
    for case in testsett:
        sporsmal = case["sporsmal"]
        forventet = case["forventet"]
        forventet_kilde = case.get("kilde")  # valgfritt felt i testsettet
        try:
            nullstill_samtale()  # hver test skal starte "blankt", uavhengig av forrige spørsmål
            data = sporr_systemet(sporsmal)
            faktisk = data["svar"]
            faktisk_kilde = data.get("kilde") or ""
            advarsel = data.get("advarsel")

            tekst_matcher = er_match(forventet, faktisk)

            kilde_stemmer = None
            if forventet_kilde:
                kilde_stemmer = forventet_kilde.lower() == faktisk_kilde.lower()

            # Testen består KUN hvis: teksten matcher, systemet ikke selv
            # varslet om mulig hallusinasjon, OG (hvis oppgitt) kilden stemmer.
            bestatt = tekst_matcher and not advarsel and (kilde_stemmer is not False)

            resultater.append(TestResultat(
                sporsmal, forventet, faktisk, bestatt,
                advarsel=advarsel, kilde=faktisk_kilde,
                forventet_kilde=forventet_kilde, kilde_stemmer=kilde_stemmer,
            ))
        except Exception as e:
            resultater.append(
                TestResultat(sporsmal, forventet, None, False, feilmelding=str(e))
            )
    return resultater


def skriv_rapport(resultater: list[TestResultat]) -> int:
    antall_ok = sum(r.bestatt for r in resultater)
    total = len(resultater)

    print(f"\n{'='*60}")
    print(f"TESTRAPPORT: {antall_ok}/{total} bestått")
    print(f"{'='*60}\n")

    for r in resultater:
        status = "✓ OK  " if r.bestatt else "✗ FEIL"
        print(f"{status} | Spørsmål:  {r.sporsmal}")
        print(f"        | Forventet: {r.forventet}")
        if r.feilmelding:
            print(f"        | Feil:      {r.feilmelding}")
        else:
            print(f"        | Faktisk:   {r.faktisk}")
            if r.kilde:
                print(f"        | Kilde:     {r.kilde}")
            if r.advarsel:
                print(f"        | ⚠ Advarsel: {r.advarsel}")
        print("-" * 60)

    print(f"\nResultat: {antall_ok}/{total} bestått "
          f"({round(100 * antall_ok / total)}%)\n")

    # Returner exit-kode: 0 = alt OK, 1 = minst én feilet (nyttig i CI)
    return 0 if antall_ok == total else 1


# ---------------------------------------------------------------------------
# 4) KJØR SOM SCRIPT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Bruk: python test_harness.py <sti-til-testsett.json>")
        sys.exit(2)

    resultater = kjor_tester(sys.argv[1])
    exit_code = skriv_rapport(resultater)
    sys.exit(exit_code)
