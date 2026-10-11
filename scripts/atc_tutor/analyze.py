"""Measurement and failure analysis for a judged tutor arm on the development splits.

rejudge:  a second, independent Opus judgment of every verified row (same frozen judge
          prompt, schema and reference) to measure judge test-retest agreement.
classify: Opus assigns each failure an error taxonomy and assesses whether the
          judgment itself holds (judge right, too strict, reference or gold-text problem).
report:   writes analysis/<arm>/report.md with retest agreement and failure breakdowns.
audit:    writes a stratified human-audit sample (failures and passes) as JSONL.

Never reads final-test artifacts: the splits are devpool, devdiag and selection only.

Usage:
  analyze.py rejudge --arm v2i328-short
  analyze.py classify --arm v2i328-short
  analyze.py report --arm v2i328-short
  analyze.py audit --arm v2i328-short --size 100
"""

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict

from bench import BENCH, CONTRACT, JUDGE_SYSTEM, JUDGE_TOOL, REFERENCE_SCHEMA, RUBRIC, correct
from opus import OUT, call_tool, client, ledger_summary

SPLITS = ("devpool", "devdiag", "selection")
ANALYSIS = OUT / "analysis"

CATEGORIES = {
    "callsign": "callsign or registration identity wrong, dropped letters, or invented letters",
    "value": "an operational value is wrong (heading, level, altitude, frequency, squawk, runway, speed, QNH, time)",
    "gloss": "value right but its meaning glossed wrongly (flight level as feet, compass direction, units)",
    "speech_act": "wrong kind of transmission: instruction vs readback vs request vs report, expectation vs clearance",
    "speaker": "wrong speaker attribution (controller vs pilot)",
    "omission": "a required proposition is simply left out",
    "invented_context": "adds an unstated reason, purpose, location, phase of flight or consequence",
    "garbled_guess": "the input is unclear or fragmentary and the answer states a confident reading instead of hedging",
    "condition_negation": "a condition, restriction, negation or correction is lost or inverted",
    "other": "anything else",
}
VALIDITY = {
    "judge_right": "the answer really fails the rubric as judged",
    "judge_too_strict": "the answer is acceptable under the contract; the judge penalized wording, an optional item or a faithful reading",
    "reference_problem": "the reference requires something wrong, or misses an acceptable reading or speaker",
    "gold_unreliable": "the gold transcript itself is too fragmentary or ambiguous for any single correct answer",
}
FIXES = {
    "data_targets": "training targets of this kind likely teach the error; fix or filter them",
    "more_coverage": "the pattern is under-represented; add varied examples",
    "hedging": "the model should express uncertainty instead of guessing",
    "number_parsing": "spoken-number parsing (digits read as a whole number, dropped or merged digits)",
    "eval_fix": "no model change needed; the evaluation is at fault",
    "unclear": "no clear model-side fix",
}
CLASSIFY_TOOL = {"name": "classify", "input_schema": {
    "type": "object", "required": ["validity", "primary", "categories", "fix", "note"],
    "properties": {
        "validity": {"enum": list(VALIDITY)},
        "primary": {"enum": list(CATEGORIES)},
        "categories": {"type": "array", "items": {"enum": list(CATEGORIES)}},
        "fix": {"enum": list(FIXES)},
        "note": {"type": "string", "description": "One sentence: what exactly went wrong."},
    }}}


def table(d):
    return "\n".join(f"- `{k}`: {v}" for k, v in d.items())


CLASSIFY_SYSTEM = f"""You analyze failures of a small ATC phrase tutor model so its
developers can decide how to improve it. Each case has the gold transmission text, the
evaluation reference, the model's answer and a grader's verdict. Contract the tutor and
grader follow:

{CONTRACT}

{RUBRIC}

First decide whether the verdict holds (validity):
{table(VALIDITY)}

Then name the primary error category and every category that applies:
{table(CATEGORIES)}

Finally the most promising fix:
{table(FIXES)}

Be strict and concrete. If the verdict does not hold, still name the category the grader
penalized."""


def jsonl(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def cohort(arm):
    """Verified rows of every development split with the arm's judged output."""
    rows = []
    for split in SPLITS:
        refs = {r["id"]: r for r in jsonl(BENCH / f"references-{split}.jsonl")}
        judged = jsonl(BENCH / f"judged-{split}-{arm}.jsonl")
        if not judged:
            continue
        for record in judged:
            ref = refs[record["id"]]
            if ref["verified"]:
                rows.append({"split": split, "ref": ref, "record": record})
    return rows


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def rejudge(args):
    out = ANALYSIS / args.arm / "rejudge.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in jsonl(out)}
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(row):
        ref, record = row["ref"], row["record"]
        async with gate:
            verdict = await call_tool(api, stage="analysis_rejudge", system=JUDGE_SYSTEM, tool=JUDGE_TOOL, user=json.dumps({
                "transmission": ref["text"],
                "reference": {k: ref[k] for k in REFERENCE_SCHEMA["required"]},
                "answer": record["answer"]}))
        append(out, {"id": record["id"], "split": row["split"], "answer": record["answer"], "judge": verdict})

    todo = [r for r in cohort(args.arm) if r["record"]["id"] not in done and not r["record"]["error"]]
    await asyncio.gather(*(run(r) for r in todo))
    print(out, len(jsonl(out)), "ledger:", json.dumps(ledger_summary().get("analysis_rejudge")))


async def classify(args):
    out = ANALYSIS / args.arm / "failures.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in jsonl(out)}
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(row):
        ref, record = row["ref"], row["record"]
        async with gate:
            result = await call_tool(api, stage="analysis_classify", system=CLASSIFY_SYSTEM, tool=CLASSIFY_TOOL,
                                     user=json.dumps({
                                         "transmission": ref["text"],
                                         "reference": {k: ref[k] for k in REFERENCE_SCHEMA["required"]},
                                         "answer": record["answer"],
                                         "verdict": record["judge"]}))
        append(out, {"id": record["id"], "split": row["split"], "transmission": ref["text"],
                     "answer": record["answer"], "judge": record["judge"], **result})

    todo = [r for r in cohort(args.arm) if not correct(r["record"]) and r["record"]["id"] not in done]
    await asyncio.gather(*(run(r) for r in todo))
    print(out, len(jsonl(out)), "ledger:", json.dumps(ledger_summary().get("analysis_classify")))


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def report(args):
    rows = cohort(args.arm)
    base = ANALYSIS / args.arm
    lines = [f"# Failure analysis: `{args.arm}` on development splits", "",
             "Verified rows of devpool, devdiag and selection. Final-test artifacts are not read. "
             "Classification and re-judging are by Opus, the same model family as the teacher and "
             "the judge, so they measure consistency, not independent truth.", ""]
    first = {r["record"]["id"]: correct(r["record"]) for r in rows}
    by_split = defaultdict(lambda: [0, 0])
    for r in rows:
        by_split[r["split"]][0] += first[r["record"]["id"]]
        by_split[r["split"]][1] += 1
    lines += ["## First judgment", ""] + [f"- {s}: correct {pct(c, n)}" for s, (c, n) in by_split.items()]

    second = {r["id"]: correct(r) for r in jsonl(base / "rejudge.jsonl")}
    if second:
        common = sorted(set(first) & set(second))
        agree = sum(first[i] == second[i] for i in common)
        flips_up = sum(not first[i] and second[i] for i in common)
        flips_down = sum(first[i] and not second[i] for i in common)
        both = sum(first[i] and second[i] for i in common)
        lines += ["", "## Judge test-retest (same prompt, same reference, independent call)", "",
                  f"- Rows: {len(common)}; same correct/incorrect verdict: {pct(agree, len(common))}",
                  f"- First correct: {pct(sum(first[i] for i in common), len(common))}; "
                  f"second correct: {pct(sum(second[i] for i in common), len(common))}",
                  f"- Fail→pass flips: {flips_up}; pass→fail flips: {flips_down}; correct in both: {pct(both, len(common))}",
                  f"- Rows judged incorrect at least once: {len(common) - both}"]

    failures = jsonl(base / "failures.jsonl")
    if failures:
        n = len(failures)
        validity = Counter(f["validity"] for f in failures)
        lines += ["", f"## Failures classified: {n}", "", "### Does the verdict hold?", ""]
        lines += [f"- `{k}`: {pct(validity[k], n)}" for k in VALIDITY]
        real = [f for f in failures if f["validity"] == "judge_right"]
        lines += ["", f"### Primary category (verdict holds, {len(real)} rows)", ""]
        primary = Counter(f["primary"] for f in real)
        lines += [f"- `{k}`: {pct(v, len(real))}" for k, v in primary.most_common()]
        lines += ["", "### Any category (verdict holds)", ""]
        anyc = Counter(c for f in real for c in set(f["categories"]) | {f["primary"]})
        lines += [f"- `{k}`: {pct(v, len(real))}" for k, v in anyc.most_common()]
        lines += ["", "### Suggested fix (all failures)", ""]
        lines += [f"- `{k}`: {pct(v, n)}" for k, v in Counter(f["fix"] for f in failures).most_common()]
        verified = len(rows)
        adjusted = sum(first.values()) + validity["judge_too_strict"] + validity["reference_problem"] + validity["gold_unreliable"]
        lines += ["", "### Implied ceiling", "",
                  f"- Correct as judged: {pct(sum(first.values()), verified)}",
                  f"- Counting failures the classifier says are evaluation problems as correct: {pct(adjusted, verified)}",
                  "", "### Examples (verdict holds), first three per primary category", ""]
        shown = Counter()
        for f in sorted(real, key=lambda f: f["id"]):
            if shown[f["primary"]] < 3:
                shown[f["primary"]] += 1
                lines += [f"- **{f['primary']}** — “{f['transmission']}” → “{f['answer']}” — {f['note']}"]
        lines += ["", "### Examples where the verdict does not hold", ""]
        for f in sorted((f for f in failures if f["validity"] != "judge_right"), key=lambda f: f["id"])[:12]:
            lines += [f"- **{f['validity']}** — “{f['transmission']}” → “{f['answer']}” — {f['note']}"]
    path = base / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


def audit(args):
    """Stratified human-audit sample: failures by classifier validity, plus passes."""
    rows = {r["record"]["id"]: r for r in cohort(args.arm)}
    failures = jsonl(ANALYSIS / args.arm / "failures.jsonl")
    rng = random.Random(20261010)
    by_validity = defaultdict(list)
    for f in failures:
        by_validity[f["validity"]].append(f["id"])
    passes = sorted(i for i, r in rows.items() if correct(r["record"]))
    n_fail = args.size * 6 // 10
    picked = []
    for v, ids in sorted(by_validity.items()):
        share = max(1, round(n_fail * len(ids) / len(failures)))
        picked += rng.sample(sorted(ids), min(share, len(ids)))
    picked += rng.sample(passes, args.size - len(picked))
    rng.shuffle(picked)
    classified = {f["id"]: f for f in failures}
    out = ANALYSIS / args.arm / "audit-sample.jsonl"
    with out.open("w") as f:
        for i in picked:
            r = rows[i]
            f.write(json.dumps({"id": i, "split": r["split"], "transmission": r["ref"]["text"],
                                "reference": {k: r["ref"][k] for k in REFERENCE_SCHEMA["required"]},
                                "answer": r["record"]["answer"], "judge": r["record"]["judge"],
                                "judged_correct": correct(r["record"]),
                                "classifier": {k: classified[i][k] for k in ("validity", "primary", "note")}
                                if i in classified else None}) + "\n")
    print(out, len(picked))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("rejudge", "classify", "report", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--concurrency", type=int, default=16)
        p.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    if args.cmd in ("rejudge", "classify"):
        asyncio.run(globals()[args.cmd](args))
    else:
        globals()[args.cmd](args)


if __name__ == "__main__":
    main()
