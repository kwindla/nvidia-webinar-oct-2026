"""Own-error repair slice for the ATC tutor (v3), following docs/reviews/atc-tutor-v2-astra.md.

reserve: split dev rows into devdiag (~150 rows by recording component, never mined or
         repaired; an independent diagnostic) and devpool (mined).
repair:  for each devpool row the tuned model got wrong (bench.py judged-devpool-<arm>),
         a candidate-blind call writes a terse faithful explanation from the transmission
         alone; check.py and an adjudicator against the frozen reference must pass.
         Writes repairs-<arm>.jsonl (SFT rows) and dpo-pairs-<arm>.jsonl, appending per
         item (resumable); quotas per failure class; infrastructure failures and
         disputed references are never mined.

Usage:
  uv run python scripts/atc_tutor/mine.py reserve
  uv run python scripts/atc_tutor/bench.py references --split devpool
  uv run python scripts/atc_tutor/bench.py run --split devpool --arm v2-iNNN:short --url URL --checkpoint PATH
  uv run python scripts/atc_tutor/bench.py judge --split devpool
  uv run python scripts/atc_tutor/mine.py repair --arm v2-iNNN-short
"""

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict

from bench import BENCH, REF_CONFIG_HASH, REFERENCE_SCHEMA, correct, judge_key, jsonl
from check import check_row, normalize
from opus import OUT, ROOT, call_tool, client, ledger_summary

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
REPAIR_TOOL = {"name": "explain", "input_schema": {
    "type": "object", "required": ["explanation"],
    "properties": {"explanation": {"type": "string"}}}}
REPAIR_SYSTEM = f"""You are the ATC phrase tutor described in this contract and write the
ideal answer for one transmission:

{CONTRACT}

Write one plain spoken sentence that conveys every operational fact in the visible
words (speech act, speaker only when evident, values with their meaning, conditions,
expectation vs clearance, negation, corrections) and nothing else. Be terse: add a gloss
only when it explains a term. Never expand an abbreviated callsign into a registration,
never guess garbled values, and derive any conversion exactly."""
ADJUDICATE_TOOL = {"name": "adjudicate", "input_schema": {
    "type": "object", "required": ["correct", "problems"],
    "properties": {"correct": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}}
ADJUDICATE_SYSTEM = f"""You independently verify an ATC tutor answer. Contract:

{CONTRACT}

Given the transmission, its reference (required propositions, optional items,
alternative readings, acceptable speakers) and an answer, mark correct only if the
answer conveys every required proposition, makes no unsupported operational claim, and
uses an acceptable speaker."""


def reserve(seed):
    rows = [json.loads(l) for l in (OUT / "split.jsonl").open()]
    dev = [r for r in rows if r["split"] == "dev"]
    components = sorted({r["component"] for r in dev})
    rng = random.Random(seed)
    rng.shuffle(components)
    diag, count = set(), 0
    for component in components:
        if count >= 150:
            break
        diag.add(component)
        count += sum(1 for r in dev if r["component"] == component)
    out = [{**r, "partition": "devdiag" if r["component"] in diag else "devpool"} for r in dev]
    path = OUT / "dev-partition.jsonl"
    if path.exists():
        raise SystemExit("dev-partition.jsonl exists; the partition is frozen")
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(Counter(r["partition"] for r in out))


REGISTRATION_ONLY = re.compile(r"regist|prefix|letters|callsign", re.I)
QUOTAS = {"unsupported": 160, "speaker": 80, "omission": 120, "registration_only": 40}


def classify(judge):
    unsupported = judge["unsupported"]
    if unsupported and all(REGISTRATION_ONLY.search(u) for u in unsupported) and not judge["missing_required"]:
        return "registration_only"
    if unsupported:
        return "unsupported"
    return "speaker" if not judge["speaker_acceptable"] else "omission"


async def repair(arm, concurrency):
    partition = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "dev-partition.jsonl").open()}
    devpool = {i for i, r in partition.items() if r["partition"] == "devpool"}
    ref_rows = jsonl(BENCH / "references-devpool.jsonl")
    refs = {r["id"]: r for r in ref_rows}
    if len(refs) != len(ref_rows) or set(refs) != devpool:
        raise SystemExit(f"devpool references incomplete: {len(refs)} of {len(devpool)}")
    if any(r.get("ref_config") != REF_CONFIG_HASH for r in ref_rows):
        raise SystemExit("devpool references were written with a different configuration")
    judged_rows = jsonl(BENCH / f"judged-devpool-{arm}.jsonl")
    outputs = {o["id"]: o for o in jsonl(BENCH / f"outputs-devpool-{arm}.jsonl")}
    judged = {r["id"]: r for r in judged_rows}
    if len(judged) != len(judged_rows) or set(judged) != devpool:
        raise SystemExit(f"judged-devpool-{arm} incomplete or duplicated")
    if len({json.dumps(r["identity"], sort_keys=True) for r in judged_rows}) != 1:
        raise SystemExit("mixed arm identities")
    if any(r["key"] != judge_key(outputs[i], refs[i]) for i, r in judged.items()):
        raise SystemExit("stale judgments: re-run bench.py judge")

    stats = Counter(judged=len(judged))
    candidates = defaultdict(list)
    for row_id, row in sorted(judged.items()):
        if not refs[row_id]["verified"]:
            stats["skip_disputed_reference"] += 1
            continue
        if row["error"] or not row["answer"]:
            stats["skip_infrastructure"] += 1  # not a semantic error; never a DPO rejected answer
            continue
        if correct(row):
            continue
        candidates[classify(row["judge"])].append(row)
    rng = random.Random(20261010)
    wrong = []
    for failure, rows in sorted(candidates.items()):
        rng.shuffle(rows)
        stats[f"wrong:{failure}"] = len(rows)
        wrong += [(failure, r) for r in rows[: QUOTAS[failure]]]

    out_path, pair_path = OUT / f"repairs-{arm}.jsonl", OUT / f"dpo-pairs-{arm}.jsonl"
    done = {json.loads(l)["seed_id"] for l in out_path.open()} if out_path.exists() else set()
    tried_path = OUT / f"repairs-{arm}.tried"
    tried = set(tried_path.read_text().split()) if tried_path.exists() else set()
    api, gate, lock = client(), asyncio.Semaphore(concurrency), asyncio.Lock()

    async def run(failure, row):
        ref = refs[row["id"]]
        async with gate:
            fixed = (await call_tool(api, stage="mine_repair", system=REPAIR_SYSTEM, tool=REPAIR_TOOL,
                                     user=json.dumps({"transmission": ref["text"]})))["explanation"].strip()
            check = check_row({"utterance": ref["text"], "explanation": fixed, "values_used": [], "bundle": {}}, frozenset())
            verdict = None
            if check["status"] != "reject" and normalize(fixed) != normalize(row["answer"]):
                verdict = await call_tool(api, stage="mine_adjudicate", system=ADJUDICATE_SYSTEM, tool=ADJUDICATE_TOOL,
                                          user=json.dumps({"transmission": ref["text"],
                                                           "reference": {k: ref[k] for k in REFERENCE_SCHEMA["required"]},
                                                           "answer": fixed}))
        async with lock:
            with tried_path.open("a") as f:
                f.write(row["id"] + "\n")
            if check["status"] == "reject":
                stats["repair_check_reject"] += 1
                return
            if verdict is None or not verdict["correct"] or verdict["problems"]:
                stats["repair_adjudicator_reject"] += 1
                return
            stats[f"repaired:{failure}"] += 1
            with out_path.open("a") as f:
                f.write(json.dumps({"slice": "repair", "seed_id": row["id"], "utterance": ref["text"],
                                    "explanation": fixed, "failure": failure, "ref_config": REF_CONFIG_HASH,
                                    "arm_identity": row["identity"],
                                    "audit": {"verdict": "pass", "adjudication": verdict}, "check": check}) + "\n")
            with pair_path.open("a") as f:
                f.write(json.dumps({"id": row["id"], "transmission": ref["text"], "chosen": fixed,
                                    "rejected": row["answer"], "failure": failure, "judge": row["judge"],
                                    "arm_identity": row["identity"]}) + "\n")

    async def safe(failure, row):
        try:
            await run(failure, row)
        except Exception as error:  # noqa: BLE001 - resumable; report and continue
            stats["error"] += 1
            print("repair failed:", row["id"], type(error).__name__, error)

    await asyncio.gather(*(safe(f, r) for f, r in wrong if r["id"] not in done and r["id"] not in tried))
    print(dict(stats))
    print("ledger:", json.dumps(ledger_summary()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reserve", "repair"])
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.command == "reserve":
        reserve(args.seed)
    else:
        asyncio.run(repair(args.arm, args.concurrency))


if __name__ == "__main__":
    main()
