"""Model-only audit of checked rows (Claude Opus 5.5, high effort).

Pass 1 (critic, every non-rejected row): reconstruct the meaning from the visible
utterance alone, then compare the candidate; verdict pass / fix / drop. A fix is new
data and must pass check.py again (max two repairs).
Pass 2 (adjudicator, disputed or high-consequence rows): an independent call that
sees only the utterance and the final candidate and cites the governing document
for disputed terms. Unresolved rows are dropped.

Usage: uv run python scripts/atc_tutor/critic.py [--concurrency 12]
Input checked.jsonl -> audited.jsonl (resumable by utterance).
"""

import argparse
import asyncio
import json
import re
from collections import Counter

from check import check_row, normalize
from opus import OUT, ROOT, call_tool, client, ledger_summary

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
HIGH_CONSEQUENCE = re.compile(
    r"\b(seven seven zero zero|seven six zero zero|seven five zero zero|mayday|pan pan|"
    r"expect|after|negative|not|correction|hold short|line up|go around|cleared|unless|until|behind)\b"
)

CRITIC_TOOL = {"name": "critique", "input_schema": {
    "type": "object", "required": ["reconstruction", "omissions", "unsupported", "speaker_ok", "verdict", "fixed_explanation"],
    "properties": {
        "reconstruction": {"type": "string", "description": "Your own reading of the visible utterance, written before judging."},
        "omissions": {"type": "array", "items": {"type": "string"}},
        "unsupported": {"type": "array", "items": {"type": "string"}},
        "speaker_ok": {"type": "boolean"},
        "verdict": {"enum": ["pass", "fix", "drop"]},
        "fixed_explanation": {"type": "string", "description": "Corrected explanation if verdict is fix, else empty."},
    }}}
ADJUDICATOR_TOOL = {"name": "adjudicate", "input_schema": {
    "type": "object", "required": ["correct", "problems", "citation"],
    "properties": {
        "correct": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "citation": {"type": "string", "description": "Governing source for any disputed term (ICAO Doc 4444 / Doc 9432 / FAA JO 7110.65 section), or empty."},
    }}}

CRITIC_SYSTEM = f"""You audit training data for an ATC phrase tutor against this contract:

{CONTRACT}

You receive a radio transmission (the only thing the student model will see) and a
candidate explanation. First write your own reconstruction of what the visible
transmission means, without looking at the candidate's wording. Then compare: list
material omissions (speech act, conditions, expectations vs clearances, negation,
corrections, values) and unsupported claims (anything not in the visible words,
including guessed repairs of garbled values). Check the speaker attribution.
Also judge the transmission itself: it must be realistic radio phraseology (correct
units, e.g. wind and speed in knots; plausible values; standard word use). If it is not,
the verdict is drop, because the student would learn wrong phraseology.
Verdict: pass if fully faithful and contract-compliant; fix if a corrected one-sentence
explanation can be written (provide it); drop if the transmission is unusable or
unrealistic."""

ADJUDICATOR_SYSTEM = f"""You independently verify one explanation from an ATC phrase
tutor dataset. Contract:

{CONTRACT}

Decide whether the explanation is a correct, faithful, contract-compliant reading of
the visible transmission. Judge standard phraseology by ICAO Doc 4444 and Doc 9432
(FAA JO 7110.65 for US phrasing); cite the relevant source for any disputed term.
Be strict about cleared vs expect, conditions, negation, corrections, runway sides
and values, and about invented details."""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--input", default="checked.jsonl")
    parser.add_argument("--output", default="audited.jsonl")
    args = parser.parse_args()
    real = frozenset(normalize(json.loads(l)["text"]) for l in (OUT / "split.jsonl").open())
    rows = [json.loads(l) for l in (OUT / args.input).open()]
    for row in rows:
        row.setdefault("check", {"status": "pass", "reasons": [], "review": []})
    path = OUT / args.output
    done = {json.loads(l)["utterance"] for l in path.open()} if path.exists() else set()
    todo = [r for r in rows if r["check"]["status"] != "reject" and r["utterance"] not in done]
    api = client()
    gate = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = Counter()

    async def audit(row):
        history = []
        explanation = row["explanation"]
        async with gate:
            for _ in range(3):
                review = await call_tool(api, stage="critic", system=CRITIC_SYSTEM, tool=CRITIC_TOOL,
                                         user=json.dumps({"transmission": row["utterance"], "candidate": explanation}))
                history.append({"explanation": explanation, **review})
                if review["verdict"] != "fix":
                    break
                fixed = review["fixed_explanation"].strip()
                recheck = check_row({**row, "explanation": fixed}, real)
                history[-1]["recheck"] = recheck
                if recheck["status"] == "reject":
                    review = {**review, "verdict": "drop"}
                    break
                explanation = fixed
            verdict = review["verdict"]
            if verdict == "fix":
                verdict = "drop"  # repair cap reached
            adjudication = None
            disputed = len(history) > 1 or row["check"]["status"] == "needs_review"
            if verdict == "pass" and (disputed or HIGH_CONSEQUENCE.search(row["utterance"])):
                adjudication = await call_tool(api, stage="adjudicate", system=ADJUDICATOR_SYSTEM, tool=ADJUDICATOR_TOOL,
                                               user=json.dumps({"transmission": row["utterance"], "explanation": explanation}))
                if not adjudication["correct"]:
                    verdict = "drop"
        record = {**row, "original_explanation": row["explanation"], "explanation": explanation,
                  "audit": {"verdict": verdict, "history": history, "adjudication": adjudication}}
        stats[verdict] += 1
        stats["fixed"] += explanation != row["explanation"] and verdict == "pass"
        stats["adjudicated"] += adjudication is not None
        async with lock:
            with path.open("a") as out:
                out.write(json.dumps(record) + "\n")

    async def safe(row):
        try:
            await audit(row)
        except Exception as error:  # noqa: BLE001 - resumable
            stats["error"] += 1
            print("audit failed:", type(error).__name__, error)

    await asyncio.gather(*(safe(r) for r in todo))
    print(dict(stats))
    print("ledger:", json.dumps(ledger_summary()))


if __name__ == "__main__":
    asyncio.run(main())
