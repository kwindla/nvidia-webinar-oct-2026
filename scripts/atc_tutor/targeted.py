"""v4 targeted slice: headings (and heading/runway minimal pairs) from code-built specs.

The 20-turn live smoke showed the v2 model glossing "heading two seven zero" as
"twenty-seven degrees", a heading/runway confusion; the real dataset has only 18
heading rows. Code samples every value and the structure; Opus 5.5 (contract v1.2)
writes the spoken transmission and the explanation. Rows then go through check.py
and critic.py like every other slice.

Usage:
  uv run python scripts/atc_tutor/targeted.py --count 250
  uv run python scripts/atc_tutor/check.py --input generated-targeted.jsonl --output checked-targeted.jsonl
  uv run python scripts/atc_tutor/critic.py --input checked-targeted.jsonl --output audited-targeted.jsonl
"""

import argparse
import asyncio
import json
import random

from generate import TOOL
from opus import OUT, ROOT, call_tool, client, items, ledger_summary
from values import Sampler, compass, spoken_digits, words_int

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
VERBS = ["turn left heading", "turn right heading", "fly heading", "continue heading", "maintain heading",
         "heading", "turn left onto heading", "after departure fly heading"]
EXTRAS = [None, None, "climb", "descend", "vectors for the ils approach", "flight level"]
SYSTEM = f"""You write training data for an ATC phrase tutor. Follow this contract exactly:

{CONTRACT}

You receive a structured spec with every value already chosen. Write one realistic
lowercase spoken radio transmission that contains exactly these elements (numbers as
words in radio form, as given in the spec's spoken fields) in a natural order for the
given speaker, then the tutor's one-sentence explanation. Headings are degrees
magnetic: "heading two seven zero" is two hundred seventy degrees, roughly west; a
runway number is a runway name, never a heading. When the spec contains both, keep
them distinct. Use the spec's compass word for the heading if you mention one."""


def build_specs(count, seed):
    rng = random.Random(seed)
    specs = []
    for index in range(count):
        sampler = Sampler(f"targeted:{seed}:{index}")
        dialect = rng.choice(["icao_europe", "icao_europe", "icao_australia", "faa"])
        bundle = sampler.sample(dialect)
        if rng.random() < 0.15:
            bundle["callsign"] = sampler.abbreviated_callsign()
        heading = bundle["heading"]
        spec = {"speaker": rng.choice(["controller", "controller", "pilot"]),
                "verb": rng.choice(VERBS), "heading": heading, "callsign": bundle["callsign"],
                "compass": compass(heading["value"])}
        if rng.random() < 0.35:
            # Minimal pair: a runway whose number shares the heading's leading digits.
            number = max(1, min(36, round(heading["value"] / 10)))
            side = rng.choice(["", "", "L", "R"])
            spec["runway"] = {"value": f"{number:02d}{side}",
                              "spoken": "runway " + spoken_digits(f"{number:02d}") +
                                        {"L": " left", "R": " right", "": ""}[side],
                              "plain": words_int(number) + {"L": " left", "R": " right", "": ""}[side]}
        extra = rng.choice(EXTRAS)
        if extra in ("climb", "descend"):
            spec["extra"] = {"instruction": extra, "altitude": bundle["altitude"]}
        elif extra == "flight level":
            spec["extra"] = {"instruction": rng.choice(["climb", "descend"]), "flight_level": bundle["flight_level"]}
        elif extra:
            spec["extra"] = {"instruction": extra}
        specs.append({"id": f"targeted-heading-{index:04d}", "dialect": dialect, "spec": spec, "bundle": bundle})
    return specs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    path = OUT / "generated-targeted.jsonl"
    done = {json.loads(l)["seed_id"] for l in path.open()} if path.exists() else set()
    api, gate, lock = client(), asyncio.Semaphore(args.concurrency), asyncio.Lock()
    specs = [s for s in build_specs(args.count, args.seed) if s["id"] not in done]

    async def run(spec):
        async with gate:
            result = await call_tool(api, stage="targeted", system=SYSTEM, tool=TOOL, user=json.dumps({
                "spec": spec["spec"], "bundles": [{"index": 0, **spec["bundle"]}], "count": 1}))
        variants = items(result, "variants")[:1]
        if not variants:
            return
        v = variants[0]
        used = ["heading"] + (["callsign"] if spec["bundle"]["callsign"]["type"] == "abbreviated" else [])
        bundle = {**spec["bundle"]}
        if "runway" in spec["spec"]:
            bundle["runway"] = spec["spec"]["runway"]
            used.append("runway")
        record = {"seed_id": spec["id"], "shape": "targeted|heading", "dialect": spec["dialect"],
                  "utterance": " ".join(v["utterance"].lower().split()), "speaker": v["speaker"],
                  "explanation": v["explanation"].strip(), "values_used": used,
                  "propositions": v["propositions"], "bundle": bundle, "slice": "targeted"}
        async with lock:
            with path.open("a") as out:
                out.write(json.dumps(record) + "\n")

    await asyncio.gather(*(run(s) for s in specs))
    print("targeted rows:", sum(1 for _ in path.open()))
    print("ledger:", json.dumps(ledger_summary()))


if __name__ == "__main__":
    asyncio.run(main())
