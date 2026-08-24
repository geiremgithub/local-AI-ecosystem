import os
import sys
import subprocess
import time
import secrets

from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------
# SIKKERHET (CSRF): Uten en hemmelig token kan EN HVILKEN SOM HELST
# nettside du besøker mens denne serveren kjører, sende skjulte
# POST-forespørsler til http://127.0.0.1:5001/start eller /stopp
# (nettlesere blokkerer ikke dette - kun det å LESE svaret). En ekstern
# side kan derfor stoppe/starte serveren din uten at du klikket noe.
#
# Løsningen: en tilfeldig token genereres hver gang denne styrings-
# serveren startes, og limes inn i HTML-siden vi selv serverer. Kun
# den ekte siden kjenner tokenen (en ekstern side kan ikke lese den,
# siden nettlesere håndhever "same-origin policy"), så /start og
# /stopp avviser alle forespørsler som mangler riktig token.
# ---------------------------------------------------------
STYRING_TOKEN = secrets.token_hex(16)


def gyldig_token():
    return request.headers.get("X-Styring-Token") == STYRING_TOKEN

# ---------------------------------------------------------
# Denne styrings-serveren er bevisst holdt superenkel og lett -
# den laster IKKE inn noen AI-modell eller database. Den eneste
# jobben den har er å starte/stoppe rag_server.py som en egen
# prosess, slik at du har et sted å trykke "Start server" fra
# nettleseren selv om hovedserveren har krasjet eller blitt
# stoppet.
#
# VIKTIG PRINSIPP: du kan ikke starte en død server fra en side
# SOM SERVERES av den samme døde serveren - denne styrings-
# siden må derfor kjøre helt uavhengig, på en annen port
# (5001), og bør helst starte automatisk sammen med Windows
# (se instruksjonene du fikk i chatten for hvordan).
# ---------------------------------------------------------

MAPPE = os.path.dirname(os.path.abspath(__file__))
RAG_SERVER_STI = os.path.join(MAPPE, "rag_server.py")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Holder styr på den kjørende hoved-server-prosessen (eller None hvis stoppet)
rag_prosess = None


def hovedserver_kjorer():
    """Sjekker om hoved-serveren faktisk kjører (og ikke bare startet, men krasjet siden)."""
    global rag_prosess
    if rag_prosess is None:
        return False
    # poll() returnerer None mens prosessen fortsatt lever, ellers exit-koden
    return rag_prosess.poll() is None


HTML_SIDE = """
<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="UTF-8">
<title>Server-styring</title>
<style>
  body { font-family: system-ui, sans-serif; background: #EDEAE2; color: #1F2937;
         display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .kort { background: #fff; border: 1px solid #D9D4C8; border-radius: 8px; padding: 2rem 2.5rem; text-align: center; min-width: 320px; }
  h1 { font-size: 1.2rem; margin-top: 0; }
  .status { font-size: 0.95rem; margin-bottom: 1.3rem; }
  .status.paa { color: #1a7f37; }
  .status.av { color: #b91c1c; }
  button { font-family: inherit; font-size: 0.95rem; padding: 0.6rem 1.2rem; border-radius: 6px; border: none;
           cursor: pointer; margin: 0 0.3rem; }
  .start { background: #1D3557; color: #fff; }
  .stopp { background: #b91c1c; color: #fff; }
  button:disabled { opacity: 0.5; cursor: default; }
  .melding { margin-top: 1rem; font-size: 0.85rem; color: #6B7280; }
</style>
</head>
<body>
<div class="kort">
  <h1>Styring av dokumentoppslag-serveren</h1>
  <div id="status" class="status">Sjekker status …</div>
  <button id="start-knapp" class="start">Start server</button>
  <button id="stopp-knapp" class="stopp">Stopp server</button>
  <div id="melding" class="melding"></div>
</div>
<script>
  const TOKEN = "__STYRING_TOKEN__";  // erstattes med ekte verdi før siden sendes
  const statusFelt = document.getElementById("status");
  const startKnapp = document.getElementById("start-knapp");
  const stoppKnapp = document.getElementById("stopp-knapp");
  const meldingFelt = document.getElementById("melding");

  async function oppdaterStatus() {
    try {
      const r = await fetch("/status");
      const data = await r.json();
      if (data.kjorer) {
        statusFelt.textContent = "Serveren kjører (http://127.0.0.1:5000)";
        statusFelt.className = "status paa";
        startKnapp.disabled = true;
        stoppKnapp.disabled = false;
      } else {
        statusFelt.textContent = "Serveren er stoppet";
        statusFelt.className = "status av";
        startKnapp.disabled = false;
        stoppKnapp.disabled = true;
      }
    } catch (e) {
      statusFelt.textContent = "Fikk ikke kontakt med styrings-serveren";
      statusFelt.className = "status av";
    }
  }

  startKnapp.addEventListener("click", async () => {
    meldingFelt.textContent = "Starter … (modellen bruker litt tid å laste)";
    await fetch("/start", { method: "POST", headers: { "X-Styring-Token": TOKEN } });
    setTimeout(oppdaterStatus, 1500);
  });

  stoppKnapp.addEventListener("click", async () => {
    meldingFelt.textContent = "Stopper …";
    await fetch("/stopp", { method: "POST", headers: { "X-Styring-Token": TOKEN } });
    setTimeout(oppdaterStatus, 1500);
  });

  oppdaterStatus();
  setInterval(oppdaterStatus, 4000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    side = HTML_SIDE.replace("__STYRING_TOKEN__", STYRING_TOKEN)
    return render_template_string(side)


@app.route("/status")
def status():
    return jsonify({"kjorer": hovedserver_kjorer()})


@app.route("/start", methods=["POST"])
def start():
    if not gyldig_token():
        return jsonify({"ok": False, "melding": "Mangler eller feil token."}), 403

    global rag_prosess
    if hovedserver_kjorer():
        return jsonify({"ok": True, "melding": "Kjører allerede."})

    # CREATE_NEW_CONSOLE gjør at rag_server.py får sitt eget terminalvindu,
    # slik at du fortsatt kan se loggene (indeksering, spørsmål, feil osv.)
    # akkurat som når du starter den manuelt.
    rag_prosess = subprocess.Popen(
        [sys.executable, RAG_SERVER_STI],
        cwd=MAPPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return jsonify({"ok": True, "melding": "Startet."})


@app.route("/stopp", methods=["POST"])
def stopp():
    if not gyldig_token():
        return jsonify({"ok": False, "melding": "Mangler eller feil token."}), 403

    global rag_prosess
    if not hovedserver_kjorer():
        return jsonify({"ok": True, "melding": "Var allerede stoppet."})

    try:
        # taskkill /T dreper hele prosesstreet (viktig siden rag_server.py
        # kjører i sitt eget konsollvindu via CREATE_NEW_CONSOLE - vanlig
        # .terminate() kan i noen tilfeller la underliggende prosesser eller
        # selve konsollvinduet henge igjen på Windows).
        subprocess.run(
            ["taskkill", "/PID", str(rag_prosess.pid), "/T", "/F"],
            capture_output=True, timeout=10,
        )
    except Exception:
        # Reserveløsning hvis taskkill ikke er tilgjengelig (f.eks. på
        # Linux/Mac under utvikling/testing)
        rag_prosess.terminate()
        try:
            rag_prosess.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rag_prosess.kill()

    rag_prosess = None
    return jsonify({"ok": True, "melding": "Stoppet."})


if __name__ == "__main__":
    print("[Styring] Kjører på http://127.0.0.1:5001 - bruk denne til å starte/stoppe hovedserveren.")
    app.run(host="127.0.0.1", port=5001, debug=False)
