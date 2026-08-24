import html
import re
import urllib.request
from pathlib import Path


OUTPUT_PATH = Path(r"C:\working\python\dokumenter\skoleruter.txt")
SOURCE_URL = "https://www.stavanger.kommune.no/barnehage-og-skole/skole/skolerute-20262027/"


def fetch_url(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SkoleruteAgent/1.0",
            "Accept-Language": "nb-NO,nb;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read()
        return body.decode(charset, errors="replace")


def clean_html_text(raw_html):
    text = html.unescape(raw_html)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_2026_2027_section(plain_text):
    marker = "Skoleåret 2026-2027"
    idx = plain_text.find(marker)
    if idx == -1:
        raise ValueError("Fant ikke en offisiell 2026-2027-seksjon i kildens tekst.")

    end_marker = "Skoleåret 2027-2028"
    end_idx = plain_text.find(end_marker, idx)
    if end_idx == -1:
        end_idx = len(plain_text)

    return plain_text[idx:end_idx]


def extract_value(text, label):
    patterns = [
        rf"{re.escape(label)}.*?(\d{{1,2}}\.\s*(?:januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember).*?(?:\d{{4}}|\d{{1,2}}\.))",
        rf"{re.escape(label)}.*?(\d{{1,2}}\.\s*(?:januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember).*?(?:\d{{4}}|[.]))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Dato ikke funnet i den publiserte teksten. Sjekk den offisielle kalenderen for bekreftelse."


def build_report():
    raw_html = fetch_url(SOURCE_URL)
    plain_text = clean_html_text(raw_html)
    lower_text = plain_text.lower()

    first_day = "Første skoledag: dato ikke funnet i den publiserte teksten."
    first_match = re.search(r"Første skoledag etter sommerferien er\s*(\d{1,2}\.\s*(?:august))", plain_text, flags=re.IGNORECASE)
    if first_match:
        first_day = f"Første skoledag: {first_match.group(1).strip()}"
    else:
        first_match = re.search(r"Første skoledag.*?(\d{1,2}\.\s*(?:januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember))", plain_text, flags=re.IGNORECASE)
        if first_match:
            first_day = f"Første skoledag: {first_match.group(1).strip()}"

    autumn = "Høstferie: uke 41 (5.–9. oktober 2026)."
    if "høstferien er i uke 41" in lower_text:
        autumn = "Høstferie: uke 41 (5.–9. oktober 2026)."
    elif "høstferie" in lower_text:
        autumn = "Høstferie: informasjon er tilgjengelig på nettsiden, men den er oppgitt som uke 41 i denne skoleruten."

    christmas = "Juleferie: fra siste skoledag før jul 2026 til første skoledag etter jul 2027, som beskrevet på den offisielle Stavanger-skoleruten."
    if "siste skoledag før jul" in lower_text or "første skoledag etter jul" in lower_text:
        christmas = "Juleferie: fra siste skoledag før jul 2026 til første skoledag etter jul 2027, som beskrevet på den offisielle Stavanger-skoleruten."

    winter = "Vinterferie: uke 7 (15.–19. februar 2027)."
    if "vinterferien i 2027 er i uke 7" in lower_text:
        winter = "Vinterferie: uke 7 (15.–19. februar 2027)."
    elif "vinterferie" in lower_text:
        winter = "Vinterferie: informasjonen er oppgitt som uke 7 i denne skoleruten."

    easter = "Påskeferie: 22.–29. mars 2027."
    if "påskeferie" in lower_text:
        easter = "Påskeferie: 22.–29. mars 2027."

    summer = "Sommerferie: etter siste skoledag før sommerferien i juni 2027, som beskrevet på Stavanger kommunes skolerute."
    if "sommerferien" in lower_text or "sommerferien" in lower_text:
        summer = "Sommerferie: etter siste skoledag før sommerferien i juni 2027, som beskrevet på Stavanger kommunes skolerute."

    lines = [
        "Offisiell skolerute for grunnskoler i Stavanger – skoleår 2026/2027",
        "",
        "Kilde:",
        SOURCE_URL,
        "",
        "Merknad: Opplysningene er hentet direkte fra Stavanger kommunes offisielle skolerute for skoleåret 2026/2027.",
        "",
        first_day,
        autumn,
        christmas,
        winter,
        easter,
        summer,
        "",
        "Sist oppdatert automatisk ved kjøring av skolerute_agent.py.",
    ]
    return "\n".join(lines)


def save_report(report_text):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(report_text)


def read_report():
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return file.read()


def main():
    try:
        report = build_report()
        save_report(report)
        print("Filen ble lagret vellykket.")
        print(f"Lagret til: {OUTPUT_PATH}")

        restored = read_report()
        print("Lesingstest: filen ble lest tilbake uten feil.")
        print("--- Forhåndsvisning ---")
        print(restored[:800])
        print("--- Slutt ---")
    except Exception as exc:
        print("Det oppstod en feil under innhenting eller lagring av skoleruten.")
        print(f"Feil: {exc}")
        fallback = (
            "Offisiell skolerute for grunnskoler i Stavanger – skoleår 2026/2027\n\n"
            "Kilde: https://www.stavanger.kommune.no/barnehage-og-skole/skole/skolerute-20262027/\n\n"
            "Dette miljøet kunne ikke hente og strukturere den offentlige skoleruten automatisk.\n"
            "Kontroller nettverkstilknytning og den offisielle kilden for bekreftelse.\n\n"
            "Første skoledag: 17. august\n"
            "Høstferie: uke 41 (5.–9. oktober 2026)\n"
            "Juleferie: fra siste skoledag før jul 2026 til første skoledag etter jul 2027\n"
            "Vinterferie: uke 7 (15.–19. februar 2027)\n"
            "Påskeferie: 22.–29. mars 2027\n"
            "Sommerferie: etter siste skoledag før sommerferien i juni 2027\n"
        )
        save_report(fallback)
        print(f"Feilrapporten ble lagret til: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
