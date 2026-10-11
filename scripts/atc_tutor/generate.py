"""Generate ATC tutor training targets from tagged dev seeds (Claude Opus 5.5).

Each call takes one seed transcript plus code-sampled value bundles and returns
variants with the same semantic shape, a different surface realization, and an
explanation that follows docs/atc-tutor-target-contract.md.

Usage: uv run python scripts/atc_tutor/generate.py --seeds N [--variants 3]
Resumable: seeds already in generated.jsonl are skipped.
"""

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict

from opus import OUT, ROOT, call_tool, client, items, ledger_summary
from values import Sampler

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
DIALECT_OF_AIRPORT = {"YSSY": "icao_australia"}

TOOL = {
    "name": "record_variants",
    "description": "Record the generated training variants.",
    "input_schema": {
        "type": "object",
        "required": ["variants"],
        "properties": {"variants": {"type": "array", "items": {
            "type": "object",
            "required": ["bundle", "utterance", "speaker", "explanation", "values_used", "propositions"],
            "properties": {
                "bundle": {"type": "integer", "description": "Index of the value bundle used."},
                "utterance": {"type": "string", "description": "Lowercase spoken radio transmission, numbers as words, no punctuation."},
                "speaker": {"enum": ["controller", "pilot", "unknown"]},
                "explanation": {"type": "string", "description": "The tutor's one-sentence explanation."},
                "values_used": {"type": "array", "items": {"type": "string"},
                                "description": "Bundle keys whose values appear in the utterance."},
                "propositions": {"type": "array", "items": {"type": "string"},
                                 "description": "Each material fact the explanation must convey, one per item."},
            }}}},
    },
}

SYSTEM = f"""You write training data for a small model that tutors people on air
traffic control radio phrases. Follow this contract exactly; it is authoritative.

{CONTRACT}

Task: you receive one real seed transmission with its clause annotation, and several
value bundles. For each bundle, write one new transmission with the SAME semantic
shape as the seed (same clause types in a natural order, same speech act and speaker
role, same modifiers such as conditions, expectations, corrections, negation and
ambiguity) but a DIFFERENT surface realization: vary word order where radio practice
allows, filler, greetings, callsign placement, and phrasing. Use the bundle's values
for every slot the shape needs, rendered in spoken radio form exactly as given in the
bundle's 'spoken' fields; you may omit values the shape does not need. Keep
disfluencies, noise markers ('spk') and truncation when the seed has them, but vary
where they occur. If the seed is a fragment, greeting or acknowledgement, write a
comparable fragment.

Then write the explanation exactly as the contract specifies: one spoken sentence,
numbers in words, meaning unpacked, speaker named only when evident, nothing
invented, uncertainty acknowledged only where the visible words are unclear. List the
values you used and the material propositions the explanation conveys."""


OTHER_TYPES = [
    (re.compile(r"continue approach"), "continue_approach"),
    (re.compile(r"hold position|stop"), "hold_position"),
    (re.compile(r"\bconfirm"), "confirm_question"),
    (re.compile(r"\bready\b"), "ready_report"),
    (re.compile(r"go ahead|break"), "go_ahead_or_break"),
    (re.compile(r"\b(ne|french|german|czech|slovak|italian)\b"), "non_english_marker"),
    (re.compile(r"noise|spk|^xt$"), "noise_marker"),
    (re.compile(r"thank|sorry"), "courtesy"),
]


def clause_type(clause):
    if clause["type"] != "other":
        return clause["type"]
    for pattern, name in OTHER_TYPES:
        if pattern.search(clause["evidence"].lower()):
            return name
    return "other"


def shape_key(tag):
    clauses = [clause_type(c) for c in tag["clauses"] if c["type"] not in {"greeting", "farewell", "station"}]
    collapsed = [t for i, t in enumerate(clauses) if i == 0 or t != clauses[i - 1]]
    modifiers = sorted({m for c in tag["clauses"] for m in c.get("modifiers", [])})
    return "|".join([tag["speaker"], *collapsed]) + ("#" + ",".join(modifiers) if modifiers else "")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=0, help="limit seeds (pilot)")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--contrast", type=int, default=0,
                        help="v2 contrast slice: N seeds with registration callsigns or flight levels, "
                             "one variant each with an abbreviated callsign and a low flight level")
    args = parser.parse_args()

    tags = [json.loads(line) for line in (OUT / "tags.jsonl").open()]
    split = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "split.jsonl").open()}
    path = OUT / ("generated-contrast.jsonl" if args.contrast else "generated.jsonl")
    done = {json.loads(l)["seed_id"] for l in path.open()} if path.exists() else set()

    rng = random.Random(args.seed)
    by_shape = defaultdict(list)
    for tag in tags:
        if split[tag["id"]]["split"] == "dev":
            by_shape[shape_key(tag)].append(tag)
    # Quota: max(3, n) targets per shape, spread across its seeds (rare shapes get
    # several variants from one seed; common shapes one variant per seed).
    variants_for = {}
    for rows in by_shape.values():
        quota = max(args.variants, len(rows))
        for i, tag in enumerate(rows):
            variants_for[tag["id"]] = quota // len(rows) + (i < quota % len(rows))
    order = [t for rows in by_shape.values() for t in rows]
    if args.contrast:
        def contrastable(tag):
            slots = [s["type"] for c in tag["clauses"] for s in c.get("slots", [])]
            return "registration" in slots or "flight_level" in slots
        order = [t for t in order if contrastable(t)]
        rng.shuffle(order)
        order = order[: args.contrast]
        variants_for = {t["id"]: 1 for t in order}
    if args.seeds:
        rng.shuffle(order)
    todo = [t for t in order if t["id"] not in done][: args.seeds or None]
    print("planned targets:", sum(variants_for[t["id"]] for t in todo))

    api = client()
    gate = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = Counter()

    async def run(tag):
        airport = split[tag["id"]]["airport"]
        dialect = DIALECT_OF_AIRPORT.get(airport, "icao_europe")
        sampler = Sampler(f"{args.seed}:{tag['id']}")
        count = variants_for[tag["id"]]
        bundles = [sampler.sample(dialect) for _ in range(count)]
        if args.contrast:
            for bundle in bundles:
                bundle["callsign"] = sampler.abbreviated_callsign()
                bundle["flight_level"] = sampler.low_flight_level()
        note = ("CONTRAST EXAMPLE: identify the aircraft only by the bundle's abbreviated callsign "
                "letters and explain it as the aircraft whose callsign ends in those letters; if the "
                "shape has a level, use the bundle's flight level and gloss it exactly (flight level "
                "N is about N hundred feet).") if args.contrast else ""
        user = json.dumps({"instructions": note,
            "seed": {"text": tag["text"], "airport": airport, "annotation": {
                k: tag[k] for k in ("speaker", "speaker_confidence", "speech_acts", "clauses", "ambiguity")}},
            "bundles": [dict(index=i, **b) for i, b in enumerate(bundles)],
        })
        async with gate:
            try:
                result = await call_tool(api, stage="generate", system=SYSTEM, user=user, tool=TOOL)
            except Exception as error:  # noqa: BLE001 - resumable
                stats["failed"] += 1
                print("seed failed:", tag["id"], type(error).__name__, error)
                return
        records = []
        for variant in items(result, "variants")[:count]:
            index = variant.get("bundle", 0)
            if not 0 <= index < len(bundles):
                stats["bad_bundle"] += 1
                continue
            utterance = re.sub(r"\s+", " ", variant["utterance"].lower()).strip()
            records.append({
                "seed_id": tag["id"], "shape": shape_key(tag), "dialect": dialect,
                "utterance": utterance, "speaker": variant["speaker"],
                "explanation": variant["explanation"].strip(),
                "values_used": variant["values_used"], "propositions": variant["propositions"],
                "bundle": bundles[index], "seed_text": tag["text"],
                "slice": "contrast" if args.contrast else "shape",
            })
        async with lock:
            with path.open("a") as out:
                out.write("".join(json.dumps(r) + "\n" for r in records))
        stats["variants"] += len(records)

    await asyncio.gather(*(run(tag) for tag in todo))
    print(f"shapes: {len(by_shape)}; seeds this run: {len(todo)}; {dict(stats)}")
    print("ledger:", json.dumps(ledger_summary()))


if __name__ == "__main__":
    asyncio.run(main())
