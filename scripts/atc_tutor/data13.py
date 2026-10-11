"""v1.3 training targets (contract docs/atc-tutor-contract-v13.md).

targets --source real       one explanation per training-side real transcript (split13 `train`)
targets --source synthetic  fresh v1.3 explanations for surviving v2 ATC inputs (shape, contrast,
                            noise), skipping every row whose seed is in a quarantined component
variants                    up to two value-swapped variants of safely transformable real parents
Each target: a writer call (Opus or Terra), a critic call from the other family (reconstructs the meaning
first, then pass / fix / drop; a fix is re-checked, at most twice), and deterministic checks.
Explanations are written from the transmission, never by polishing an old target.
Outputs (resumable): artifacts/atc-tutor/v13/targets-<source>.jsonl, drops in *-drops.jsonl.

Usage:
  data13.py targets --source real --writer opus [--limit 30]
  data13.py targets --source synthetic --writer opus
  data13.py variants
"""

import argparse
import asyncio
import json
import re

import split13
import telephony
from opus import OUT, ROOT, call_tool, client, ledger_summary

V13 = OUT / "v13"
CONTRACT = (ROOT / "docs/atc-tutor-contract-v13.md").read_text()
V2 = OUT / "export/atc-tutor-v2"
PHONETIC = ("alpha alfa bravo charlie delta echo foxtrot golf hotel india juliett juliet kilo lima mike "
            "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

WRITE_TOOL = {"name": "explain", "input_schema": {
    "type": "object", "required": ["reading", "ambiguity", "explanation"],
    "properties": {
        "reading": {"type": "string", "description": "Each clause of the transmission with its meaning."},
        "ambiguity": {"enum": ["clear", "plausible_reading", "unresolved"]},
        "explanation": {"type": "string", "description": "The tutor's one spoken sentence."},
    }}}
CRITIC_TOOL = {"name": "critique", "input_schema": {
    "type": "object", "required": ["reconstruction", "problems", "severe", "verdict", "fixed_explanation"],
    "properties": {
        "reconstruction": {"type": "string", "description": "Your own reading of the transmission, written before looking at the candidate."},
        "problems": {"type": "array", "items": {"type": "string"}},
        "severe": {"type": "boolean", "description": "Any severe error as defined in the contract."},
        "verdict": {"enum": ["pass", "fix", "drop"]},
        "fixed_explanation": {"type": "string", "description": "A corrected explanation if verdict is fix, else empty."},
    }}}

WRITER_SYSTEM = f"""You write training targets for a small voice model that tutors people on
air traffic control radio phrases. Follow this contract exactly; it overrides your habits:

{CONTRACT}

The input is one transmission as transcribed (it may contain transcription errors), plus
the checked telephony table entries that occur in it. Read it clause by clause first.
Then write the one spoken sentence the tutor should say. Use only telephony glosses from
the given table entries; for any other telephony give the callsign as heard with no
airline. Never use a trailing callsign to decide the speaker."""

CRITIC_SYSTEM = f"""You audit training targets for an ATC phrase tutor against this contract:

{CONTRACT}

First reconstruct the meaning of the transmission yourself, before reading the candidate.
Then list every contract violation in the candidate: missing or changed operational
content, invented content, speaker or mode errors (a trailing callsign is never evidence of
a pilot), callsign not as heard, airline glosses not in the given table entries,
unlicensed "probably" readings, missing "probably" where one reading is clearly most
plausible, unexplained recognizable terms of art, form problems. Verdict: pass if there are
no problems; fix if you can write a correct explanation (give it); drop if the transmission
cannot be explained faithfully."""

VARIANT_TOOL = {"name": "variants", "input_schema": {
    "type": "object", "required": ["transformable", "reason", "variants"],
    "properties": {
        "transformable": {"type": "boolean"},
        "reason": {"type": "string"},
        "variants": {"type": "array", "description": "At most two.", "items": {
            "type": "object", "required": ["transmission", "changes"],
            "properties": {"transmission": {"type": "string"},
                           "changes": {"type": "string", "description": "Which slots changed and how consistency was kept."}}}},
    }}}
VARIANT_SYSTEM = f"""You make training-input variants of real air traffic control transmissions
for a tutor model. Contract for context:

{CONTRACT}

Keep the transmission's real structure exactly: word order, fragments, hesitations,
foreign-language courtesy words, split or merged words, its speech act and its ambiguity.
Change only typed slots: callsigns (a different plausible callsign of the same kind,
spoken the same way), and operational values (runway, heading, level, altitude,
frequency, squawk, QNH, wind, speed, waypoint or taxiway names) to other realistic values
consistent with each other (runway and heading pairs, repeated values and corrections stay
consistent, frequencies stay valid). Write numbers and letters as spoken words, like the
input. If the transmission is garbled, cut off, or its values' roles are unclear, set
transformable false and give no variants. Otherwise give one or two variants."""


def jsonl(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


TABLE = None


def table_hits(text):
    global TABLE
    TABLE = TABLE or telephony.load()
    padded = f" {text} "
    hits = {}
    for k, v in TABLE.items():
        if f" {k} " not in padded:
            continue
        # A phonetic-letter word ("delta") is airline telephony only when digits follow it;
        # "delta oscar zulu tango tango" is a German registration.
        if k in PHONETIC and not re.search(rf"\b{k} (zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|"
                                           r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                                           r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", text):
            continue
        hits[k] = v
    return hits


def leading_registration(text):
    """A spoken registration run of 3+ phonetic letters at the start or end of the transmission."""
    words = text.replace("x ray", "xray").split()
    for run in (words, words[::-1]):
        letters = []
        for w in run:
            if w in PHONETIC:
                letters.append(w)
            else:
                break
        if len(letters) >= 3:
            return letters if run is words else letters[::-1]
    return None


UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def words_value(phrase):
    """Parse a cardinal in words ("one thousand seventeen", "two hundred sixty") to an int, else None."""
    total, current, seen = 0, 0, False
    for word in phrase.lower().replace("-", " ").split():
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word == "thousand":
            total, current = total + (current or 1) * 1000, 0
        elif word == "and":
            continue
        else:
            break
        seen = True
    return total + current if seen else None


NUMBER = r"((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|" \
         r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|" \
         r"thousand|and)[ -]?)+)"
RANGES = [  # (pattern capturing a number in words, low, high, label)
    (rf"{NUMBER}\s*hectopascals", 900, 1100, "QNH in hectopascals"),
    (rf"heading of {NUMBER}\s*degrees|{NUMBER}\s*degrees,? (?:about |roughly )?(?:north|south|east|west)", 0, 360, "heading"),
    (rf"flight level {NUMBER}", 0, 600, "flight level"),
]


def implausible(explanation):
    """Values stated as fact that are physically implausible; 'probably' readings are judged by the critic."""
    problems = []
    for pattern, low, high, label in RANGES:
        for match in re.finditer(pattern, explanation, re.I):
            phrase = next(g for g in match.groups() if g)
            value = words_value(phrase)
            if value is not None and not low <= value <= high:
                problems.append(f"implausible {label}: {value}; say it is unclear or give the probable reading")
    for match in re.finditer(rf"{NUMBER}point {NUMBER}\s*megahertz", explanation, re.I):
        whole = words_value(match.group(1))
        if whole is not None and not 108 <= whole <= 137:
            problems.append(f"implausible VHF frequency: {whole} megahertz")
    return problems


COMPASS16 = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast",
             "south-southeast", "south", "south-southwest", "southwest", "west-southwest", "west",
             "west-northwest", "northwest", "north-northwest"]
DIGIT = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
DIGIT["niner"] = 9


def heading_problems(transmission, text):
    """A clear spoken heading must appear as exact degrees with a compass point within one 16-point sector."""
    problems = []
    for m in re.finditer(r"\bheading ((?:zero|one|two|three|four|five|six|seven|eight|nine|niner) "
                         r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner) "
                         r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner))\b", transmission):
        value = int("".join(str(DIGIT[w]) for w in m.group(1).split()))
        if not 0 < value <= 360:
            continue
        from values import words_int
        degrees = words_int(value) + " degrees"
        if degrees not in text.lower().replace("-", " ").replace("  ", " ") and degrees.replace("-", " ") not in text.lower().replace("-", " "):
            problems.append(f"heading {m.group(1)} must be given as {degrees}")
            continue
        sector = round((value % 360) / 22.5) % 16
        allowed = {COMPASS16[(sector + d) % 16] for d in (-1, 0, 1)}
        after = text.lower().split(degrees.replace("-", " ") if degrees.replace("-", " ") in text.lower() else degrees, 1)[-1][:40]
        of = re.search(r"\b(north|south|east|west) of (?:due )?(north|south|east|west)\b", after)
        if of:
            if of.group(1) == of.group(2):
                problems.append("nonsense compass gloss")
            stated = [of.group(2)]
        else:
            stated = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", after) if w in COMPASS16]
        if stated and stated[0] not in allowed:
            problems.append(f"heading {words_int(value)} degrees is {COMPASS16[sector]}, not {stated[0]}")
    return problems


def checks(transmission, explanation):
    """Deterministic v1.3 form checks; returns a list of problems."""
    problems = []
    text = explanation.strip()
    if re.search(r"\d", text):
        problems.append("digits: write numbers in words")
    if re.search(r"[*#`\"“”]", text):
        problems.append("quotes or markdown")
    if re.search(r"[.!?;]\s+\S", text):
        problems.append("more than one sentence")
    if re.search(r"\bends? in\b", text, re.I):
        problems.append("uses 'callsign ends in'; give the callsign as heard")
    if re.search(r"\b[A-Z]{1,2}-[A-Z0-9]{2,}\b|\bN\d", text):
        problems.append("hyphenated or N-number registration")
    if len(text.split()) > 45:
        problems.append(f"too long ({len(text.split())} words)")
    problems += implausible(text)
    problems += heading_problems(transmission, text)
    reg = leading_registration(transmission)
    if reg:
        heard = " ".join(reg).replace("xray", "x-ray")
        said = re.sub(r"[^a-z\- ]", " ", text.lower().replace("x ray", "x-ray"))
        if heard not in " ".join(said.split()):
            problems.append(f"registration not as heard: expected the letters '{heard}' in order")
    return problems


async def call(family, apis, stage, system, tool, user):
    """One schema-constrained call to the given model family."""
    if family == "opus":
        return await call_tool(apis["opus"], stage=stage, system=system, tool=tool, user=user)
    from bakeoff13 import terra_tool
    return await terra_tool(apis["terra"], system, tool, user)


async def write_and_audit(apis, transmission, stage, writer, facts=None):
    critic = "terra" if writer == "opus" else "opus"
    hits = table_hits(transmission)
    extra = {"checked_facts": facts} if facts else {}
    user = json.dumps({"transmission": transmission, "telephony_table_entries": hits, **extra})
    draft = await call(writer, apis, f"{stage}_write", WRITER_SYSTEM, WRITE_TOOL, user)
    explanation, history = draft["explanation"].strip(), []
    for _ in range(3):
        problems = checks(transmission, explanation)
        review = await call(critic, apis, f"{stage}_critic", CRITIC_SYSTEM, CRITIC_TOOL, json.dumps({"transmission": transmission, "telephony_table_entries": hits, **extra,
                                                  "candidate": explanation, "deterministic_problems": problems}))
        history.append({"candidate": explanation, "problems": review["problems"] + problems,
                        "verdict": review["verdict"], "severe": review["severe"]})
        if review["verdict"] == "pass" and not problems:
            return {"explanation": explanation, "ambiguity": draft["ambiguity"], "reading": draft["reading"],
                    "telephony": hits, "history": history, "writer": writer, "critic": critic}
        if review["verdict"] == "drop":
            break
        if review["verdict"] == "fix" and review["fixed_explanation"].strip():
            explanation = review["fixed_explanation"].strip()
        elif problems:
            redo = await call(writer, apis, f"{stage}_write", WRITER_SYSTEM, WRITE_TOOL, json.dumps({"transmission": transmission, "telephony_table_entries": hits, **extra,
                                                    "previous": explanation, "fix_these": problems}))
            explanation = redo["explanation"].strip()
        else:
            break
    return {"drop": True, "history": history, "telephony": hits}


def real_inputs():
    return [{"id": r["id"], "component": r["component"], "transmission": r["text"], "source": "real"}
            for r in split13.load().values() if r["part13"] == "train"]


def synthetic_inputs():
    rows, quarantined = split13.load(), split13.quarantine()
    prov = {p["conversation_id"]: p for p in jsonl(V2 / "provenance.jsonl")}
    out = []
    for conv in jsonl(V2 / "train.chat.jsonl"):
        p = prov[conv["conversation_id"]]
        if p["slice"] not in ("shape", "contrast", "noise"):
            continue
        seed = rows.get(p.get("seed_id"))
        if seed is None or seed["component"] in quarantined:
            continue
        user = [m["content"] for m in conv["messages"] if m["role"] == "user"]
        out.append({"id": conv["conversation_id"], "component": seed["component"], "seed_id": p["seed_id"],
                    "slice": p["slice"], "transmission": user[0], "source": "synthetic"})
    return out


def core_inputs():
    return data13_jsonl_core()


def data13_jsonl_core():
    return [dict(r) for r in jsonl(V13 / "core-inputs.jsonl")]


def variant_inputs():
    return [{"id": v["id"], "component": v["component"], "parent_id": v["parent_id"],
             "transmission": v["transmission"], "source": "variant"} for v in jsonl(V13 / "variant-inputs.jsonl")]


async def targets(args):
    V13.mkdir(parents=True, exist_ok=True)
    out, drops = V13 / f"targets-{args.source}.jsonl", V13 / f"targets-{args.source}-drops.jsonl"
    done = {r["id"] for r in jsonl(out)} | {r["id"] for r in jsonl(drops)}
    items = {"real": real_inputs, "synthetic": synthetic_inputs, "variant": variant_inputs,
             "core": core_inputs}[args.source]()
    todo = [i for i in items if i["id"] not in done][: args.limit or None]
    from bakeoff13 import terra_client
    apis, gate = {"opus": client(), "terra": terra_client()}, asyncio.Semaphore(args.concurrency)

    async def run(item):
        async with gate:
            try:
                result = await write_and_audit(apis, item["transmission"], f"data13_{args.source}", args.writer,
                                               item.get("facts"))
            except Exception as error:  # infrastructure failure: leave for a resumed run
                print("error", item["id"], error)
                return
        append(drops if result.get("drop") else out, {**item, **result})

    await asyncio.gather(*(run(i) for i in todo))
    kept, dropped = len(jsonl(out)), len(jsonl(drops))
    print(f"{args.source}: kept {kept}, dropped {dropped}")
    print("ledger:", {k: round(v["usd"], 2) for k, v in ledger_summary().items() if k.startswith("data13")})


async def variants(args):
    """Variant inputs from training-side real transcripts; the variant writer skips unsafe ones."""
    out = V13 / "variant-inputs.jsonl"
    done = {r["parent_id"] for r in jsonl(out)} | {r["parent_id"] for r in jsonl(V13 / "variant-skips.jsonl")}
    parents = [r for r in real_inputs() if r["id"] not in done]
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(parent):
        async with gate:
            result = await call_tool(api, stage="data13_variant", system=VARIANT_SYSTEM, tool=VARIANT_TOOL,
                                     user=json.dumps({"transmission": parent["transmission"]}))
        kept = [v for v in result["variants"] if v["transmission"].strip() and v["transmission"] != parent["transmission"]]
        if not result["transformable"] or not kept:
            append(V13 / "variant-skips.jsonl", {"parent_id": parent["id"], "reason": result["reason"]})
            return
        for k, v in enumerate(kept[:2]):
            append(out, {"id": f"{parent['id']}~v{k + 1}", "parent_id": parent["id"], "component": parent["component"],
                         "transmission": v["transmission"].strip(), "changes": v["changes"]})

    await asyncio.gather(*(run(p) for p in parents[: args.limit or None]))
    print("variant inputs:", len(jsonl(out)), "skipped parents:", len(jsonl(V13 / "variant-skips.jsonl")))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--source", choices=["real", "synthetic", "variant", "core"], required=True)
    t.add_argument("--writer", choices=["opus", "terra"], required=True,
                   help="the critic is always the other family")
    v = sub.add_parser("variants")
    for p in (t, v):
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(targets(args) if args.cmd == "targets" else variants(args))


if __name__ == "__main__":
    main()
