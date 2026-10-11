"""Owner style pass over v1.3 targets (owner calibration notes, docs/atc-tutor-eval-v13.md).

1. Known airlines (checked telephony table): "Rex sixty one thirty four, a Regional Express
   flight" -> "Regional Express flight sixty one thirty four", and an unglossed known
   telephony ("Lot four two zero") -> "LOT Polish Airlines flight four two zero".
   Deterministic.
2. Ambiguous speaker: "the speaker gives or confirms X" -> "the speaker either gives X or
   reads it back". Also drops extra glosses ("a DHL airline", "the German en-route
   controller"). An Opus minimal-edit call; the edit must keep every number word,
   phonetic letter word and callsign word, or the original stays.
Unknown telephony stays as heard. Writes targets-<source>-r.jsonl next to each input
(real, synthetic, variant, core); mode/chat rows get their ATC-style assistant turns restyled
the same way.

Usage: restyle13.py   (after `telephony.py` has rebuilt the table)
"""

import asyncio
import json
import re
from collections import Counter

import data13
import telephony
from opus import call_tool, client, ledger_summary

V13 = data13.V13
EDIT_TOOL = {"name": "edit", "input_schema": {
    "type": "object", "required": ["explanation"], "properties": {"explanation": {"type": "string"}}}}
EDIT_SYSTEM = """You make a minimal style edit to one sentence from an ATC phrase tutor. Change only:
- a hedged speaker phrase like "gives or confirms", "gives or reads back", "issues or confirms"
  becomes "either <verb>s … or reads it back" (or "or confirms it"), keeping every value and
  clause; for example "The speaker gives or confirms a landing clearance on runway sixteen
  right for Regional Express flight sixty one thirty four" becomes "The speaker either clears
  Regional Express flight sixty one thirty four to land on runway sixteen right or reads back
  that clearance";
- remove extra descriptive glosses of stations or operators that add no operational meaning
  ("a DHL airline", "the German en-route controller").
Keep every number, letter, callsign, runway, frequency, heading, level and condition exactly.
Keep it one spoken sentence with numbers in words. If nothing needs changing, return it as is."""
KEEP = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                  r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                  r"eighty|ninety|hundred|thousand|" + "|".join(data13.PHONETIC) + r"|x-ray|left|right|center)\b",
                  re.I)


def airline_style(transmission, text, table):
    """Deterministic known-airline rewrite; returns (text, changed)."""
    changed = False
    hits = data13.table_hits(transmission)
    for spoken in sorted(table, key=len, reverse=True):
        if spoken not in hits:
            continue
        airline = table[spoken].split(",")[0]  # display name: "European Air Transport, a DHL airline" -> first part
        tele = re.escape(spoken).replace(r"\ ", r"\s+")
        # "<Tele> <rest>, a/an <Airline …> flight[ for …][,]" (appositive gloss after the callsign)
        pattern = re.compile(rf"\b{tele}\s+(?P<rest>[^,]+?),\s+an?\s+{re.escape(airline)}[^,]*?\bflight\b"
                             rf"(?:\s+for\s+\w+)?(?:,\s+an?\s+[^,]*?airline)?,?", re.I)
        new = pattern.sub(lambda m: f"{airline} flight {m.group('rest')}", text)  # both commas of the aside go
        if new == text:
            # unglossed "<Tele> <number words…>" -> "<Airline> flight <number words…>"
            pattern = re.compile(rf"\b{tele}(?=\s+(?:{KEEP.pattern[3:-3]}))", re.I)
            new = pattern.sub(f"{airline} flight", text, count=1)
        if new != text:
            text, changed = new, True
        break
    return text, changed


def needs_edit(text):
    return bool(re.search(r"\b(gives|issues|gave) or (confirms|reads back)\b|or confirms\b|a DHL airline|"
                          r"the German en-route|, the [a-z\- ]+ controller,", text))


def kept(old, new):
    return Counter(w.lower() for w in KEEP.findall(old)) == Counter(w.lower() for w in KEEP.findall(new))


async def main():
    table = telephony.load()
    api, gate = client(), asyncio.Semaphore(16)
    stats = Counter()

    async def restyle(transmission, text):
        text, changed = airline_style(transmission, text, table)
        stats["airline"] += changed
        if needs_edit(text):
            async with gate:
                result = await call_tool(api, stage="restyle13_edit", system=EDIT_SYSTEM, tool=EDIT_TOOL,
                                         user=json.dumps({"transmission": transmission, "sentence": text}))
            edited = result["explanation"].strip()
            if kept(text, edited) and not data13.checks(transmission, edited):
                stats["edited"] += edited != text
                text = edited
            else:
                stats["edit_rejected"] += 1
        return text

    for source in ("real", "synthetic", "variant", "core"):
        rows = data13.jsonl(V13 / f"targets-{source}.jsonl")
        new = await asyncio.gather(*(restyle(r["transmission"], r["explanation"]) for r in rows))
        with (V13 / f"targets-{source}-r.jsonl").open("w") as f:
            for r, text in zip(rows, new):
                f.write(json.dumps({**r, "explanation": text, "pre_restyle": r["explanation"]}) + "\n")
    for source in ("mode", "chat"):
        rows = data13.jsonl(V13 / f"targets-{source}.jsonl")
        for r in rows:
            for k, t in enumerate(r["turns"]):
                if t["role"] == "assistant" and t.get("style") == "atc":
                    user = next((u["content"] for u in reversed(r["turns"][:k]) if u["role"] == "user"), "")
                    t["content"] = await restyle(user, t["content"])
        (V13 / f"targets-{source}-r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(dict(stats), "ledger:", {k: round(v["usd"], 2) for k, v in ledger_summary().items() if k.startswith("restyle13")})


if __name__ == "__main__":
    asyncio.run(main())
