"""Checked airline telephony table for v1.3 (teacher, critic, references and judge share it).

Sources, in order:
  openflights  callsign column of OpenFlights airlines.dat (downloaded to
               artifacts/atc-tutor/external/openflights/, hash recorded)
  curated      entries confirmed by the project owner or the airline's published
               callsign, listed in CURATED with a note
A telephony word gets an airline gloss only when it is in this table, and only when
it is used as airline telephony ("delta two one" at the start of a callsign, not the
letter Delta inside a registration). Writes
artifacts/atc-tutor/telephony-v13.json, restricted to telephony that occurs in the
dataset transcripts or the generator's value pool (values.TELEPHONY keys).
"""

import csv
import hashlib
import json
import re

from opus import OUT
from values import TELEPHONY

SOURCE = OUT / "external/openflights/airlines.dat"
TABLE = OUT / "telephony-v13.json"
# Spoken telephony whose published callsign differs from OpenFlights' spelling or is missing there.
CURATED = {
    "qjet": ("QantasLink", "owner-confirmed 2026-10-10: QJet is National Jet Systems flying as QantasLink"),
    "qlink": ("QantasLink", "QantasLink (Sunstate) callsign QLINK"),
    "velocity": ("Virgin Australia", "Virgin Australia callsign VELOCITY"),
    "rex": ("Regional Express", "owner audit note 2026-10-10; Regional Express callsign REX"),
    "csa": ("Czech Airlines", "Czech Airlines callsign CSA LINES, spoken CSA"),
    "air france": ("Air France", "Air France callsign AIRFRANS, spoken Air France"),
    "swiss": ("Swiss", "SWISS callsign SWISS"),
    "lufthansa": ("Lufthansa", "Lufthansa callsign LUFTHANSA"),
    "speedbird": ("British Airways", "British Airways callsign SPEEDBIRD"),
    "qantas": ("Qantas", "Qantas callsign QANTAS"),
    "jetstar": ("Jetstar", "Jetstar callsign JETSTAR"),
    "swiss ambulance": ("Swiss Air-Ambulance", "Rega callsign SWISS AMBULANCE"),
    "quality": ("ASL Airlines Belgium", "ICAO TAY, callsign QUALITY (formerly TNT Airways)"),
    "eurotrans": ("European Air Transport, a DHL airline", "ICAO BCS, callsign EUROTRANS"),
    "nostrum": ("Air Nostrum", "Air Nostrum callsign AIR NOSTRUM, often shortened to NOSTRUM"),
    "air nostrum": ("Air Nostrum", "Air Nostrum callsign AIR NOSTRUM"),
    "skytravel": ("Smartwings", "Smartwings (formerly Travel Service) callsign SKYTRAVEL"),
    # Added 2026-10-10 at the owner's request (applied after the v1.3 candidate judging round).
    "lot": ("LOT Polish Airlines", "LOT callsign LOT"),
    "turkish": ("Turkish Airlines", "Turkish Airlines callsign TURKISH"),
    "fedex": ("FedEx", "FedEx Express callsign FEDEX"),
    "ups": ("UPS Airlines", "UPS Airlines callsign UPS"),
    "austrian": ("Austrian Airlines", "Austrian Airlines callsign AUSTRIAN"),
    "alaska": ("Alaska Airlines", "Alaska Airlines callsign ALASKA"),
    "avianca": ("Avianca", "Avianca callsign AVIANCA"),
    "japan air": ("Japan Airlines", "Japan Airlines callsign JAPANAIR, spoken Japan Air"),
    "korean air": ("Korean Air", "Korean Air callsign KOREANAIR, spoken Korean Air"),
    "air india": ("Air India", "Air India callsign AIRINDIA, spoken Air India"),
    "el al": ("El Al", "El Al callsign ELAL, spoken El Al"),
}
# OpenFlights entries known to be stale or misleading for this data.
EXCLUDE = {"smartwings", "connie", "bair"}
# Callsign column values that are ordinary ATC words or too generic to gloss safely.
GENERIC = {"jet", "travel", "team", "portugal", "rescue", "apron", "tower", "radar", "ambulance"}


def build():
    raw = SOURCE.read_bytes()
    openflights = {}
    for row in csv.reader(raw.decode("utf-8").splitlines()):
        if len(row) < 8 or row[5] in ("", "\\N"):
            continue
        callsign = re.sub(r"\s+", " ", row[5].strip().lower())
        # Only active airlines: inactive entries are often reused or stale callsigns.
        if callsign in GENERIC or callsign in EXCLUDE or len(callsign) < 3 or row[7] != "Y":
            continue
        openflights.setdefault(callsign, row[1])
    texts = " " + " ".join(json.loads(l)["text"] for l in (OUT / "split.jsonl").open()) + " "
    table = {}
    for spoken, airline in openflights.items():
        if f" {spoken} " in texts or spoken in TELEPHONY:
            table[spoken] = {"airline": airline, "source": "openflights"}
    for spoken, (airline, note) in CURATED.items():
        table[spoken] = {"airline": airline, "source": "curated", "note": note}
    TABLE.write_text(json.dumps({
        "openflights_sha256": hashlib.sha256(raw).hexdigest(),
        "entries": dict(sorted(table.items()))}, indent=1) + "\n")
    print(TABLE, len(table), "entries")


def load():
    return {k: v["airline"] for k, v in json.loads(TABLE.read_text())["entries"].items()}


if __name__ == "__main__":
    build()
