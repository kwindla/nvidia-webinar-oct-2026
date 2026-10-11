"""Benchmark ATC tutor systems on the frozen selection or final-test split (rubric v2).

references: Opus writes a reference from the gold transcript; an independent pass
            verifies it and may correct any field; a revision is verified again (max 2).
            References still disputed after that are marked and excluded from scoring.
            Written BEFORE any candidate output for the split is scored.
run:        one completion per row for an arm, recording endpoint, served model,
            checkpoint, prompt hash, decoding and finish reason (resumable per arm id).
judge:      blinded Opus scoring against the reference; cached by a hash of
            (answer, reference, judge prompt), so changed inputs are re-judged.
report:     primary metric = all REQUIRED propositions conveyed, no unsupported
            operational claim, and an acceptable speaker; errors/empty answers count
            as incorrect. Paired component bootstrap against a baseline arm, and the
            frozen success decision (>= 5 points over the baseline, unsupported-claim
            rate no worse).

Usage:
  bench.py references --split selection
  bench.py run --split selection --arm tuned-v2-i312:short --url URL --checkpoint PATH
  bench.py run --split selection --arm terra:short --kind openai
  bench.py judge --split selection
  bench.py report --split selection --baseline stock:short
"""

import argparse
import asyncio
import hashlib
import json
import random
from collections import defaultdict

from dotenv import dotenv_values
from openai import AsyncOpenAI

from export import FULL_PROMPT, SHORT_PROMPT
from opus import OUT, ROOT, call_tool, client, ledger_summary

BENCH = OUT / "bench" / "v2"
# Evaluator contract frozen at v1.1 (the version references and baseline judgments used);
# generation-side contract changes (v1.2+) must not change scoring. See docs/reviews/atc-tutor-v3-plan-astra.md H5.
CONTRACT = (ROOT / "docs/atc-tutor-eval-contract.md").read_text()
PROMPTS = {"short": SHORT_PROMPT, "full": FULL_PROMPT}
DECODING = {"temperature": 0, "max_tokens": 200}

REFERENCE_SCHEMA = {
    "type": "object",
    "required": ["speakers", "required", "optional", "alternative_readings", "acceptable_uncertainty"],
    "properties": {
        "speakers": {"type": "array", "items": {"enum": ["controller", "pilot", "speaker"]},
                     "description": "Every acceptable speaker attribution; include 'speaker' when the role is not evident."},
        "required": {"type": "array", "items": {"type": "string"},
                     "description": "Material operational propositions: speech act, every operational value and its meaning, conditions, expectation vs clearance, negation, corrections."},
        "optional": {"type": "array", "items": {"type": "string"},
                     "description": "Identity and courtesy glosses a good answer may include but need not: callsign or registration identity, station name, greetings."},
        "alternative_readings": {"type": "array", "items": {"type": "string"},
                                 "description": "Other faithful readings of ambiguous wording that must also be accepted."},
        "acceptable_uncertainty": {"type": "string"},
    },
}
REF_TOOL = {"name": "reference", "input_schema": REFERENCE_SCHEMA}
VERIFY_TOOL = {"name": "verify", "input_schema": {
    "type": "object", "required": ["correct", "problems", "corrected"],
    "properties": {"correct": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}},
                   "corrected": REFERENCE_SCHEMA}}}
JUDGE_TOOL = {"name": "judge", "input_schema": {
    "type": "object", "required": ["missing_required", "unsupported", "speaker_acceptable", "one_sentence", "plain_spoken"],
    "properties": {
        "missing_required": {"type": "array", "items": {"type": "string"}},
        "unsupported": {"type": "array", "items": {"type": "string"},
                        "description": "Operational claims not supported by the transmission: invented or changed values, wrong clearance type, guessed repairs, invented registration letters."},
        "speaker_acceptable": {"type": "boolean"},
        "one_sentence": {"type": "boolean"},
        "plain_spoken": {"type": "boolean", "description": "Plain language, numbers in words, no quoting, readback or markdown."},
    }}}

RUBRIC = """Rubric v2 (frozen before the final test): an answer is correct only if it conveys every
REQUIRED proposition (any wording; an alternative reading listed in the reference also
satisfies it), makes no unsupported operational claim, and uses one of the acceptable
speakers. OPTIONAL items (callsign/registration identity, station name, greetings) are
never required. Acknowledging uncertainty described in acceptable_uncertainty is not an
omission; proposing guessed values for garbled words is an unsupported claim."""

REF_SYSTEM = f"""You write evaluation references for an ATC phrase tutor. Contract:

{CONTRACT}

{RUBRIC}

From the transmission alone, write the reference: acceptable speakers, required
operational propositions, optional identity/courtesy items, alternative faithful
readings, and acceptable uncertainty."""
VERIFY_SYSTEM = f"""You independently check an evaluation reference for an ATC phrase
tutor against the transmission. Contract:

{CONTRACT}

{RUBRIC}

Mark correct only if every field is right and internally consistent: required items
are truly material and supported, identity glosses are optional, every faithful
reading and speaker is accepted, and nothing unsupported is required. Otherwise return
the full corrected reference."""
JUDGE_SYSTEM = f"""You grade one answer from an ATC phrase tutor. You do not know which
system wrote it. Contract:

{CONTRACT}

{RUBRIC}

Compare the answer with the transmission and the reference. List missing REQUIRED
propositions and unsupported operational claims, and say whether the speaker
attribution is one of the acceptable speakers."""


REF_CONFIG_HASH = hashlib.sha256(json.dumps(
    [REF_SYSTEM, VERIFY_SYSTEM, REF_TOOL, VERIFY_TOOL, "claude-opus-5-5", "high"]).encode()).hexdigest()[:16]


def rows_for(split):
    """Frozen splits from split.jsonl; devdiag/devpool are the dev partitions made by mine.py reserve."""
    if split in ("devdiag", "devpool"):
        rows = [json.loads(l) for l in (OUT / "dev-partition.jsonl").open()]
        return [r for r in rows if r["partition"] == split]
    rows = [json.loads(l) for l in (OUT / "split.jsonl").open()]
    return [r for r in rows if r["split"] == split]


def jsonl(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


async def references(args):
    BENCH.mkdir(parents=True, exist_ok=True)
    path = BENCH / f"references-{args.split}.jsonl"
    existing = jsonl(path)
    if any(r.get("ref_config") != REF_CONFIG_HASH for r in existing):
        raise SystemExit(f"{path.name} was written with a different reference configuration")
    done = {r["id"] for r in existing}
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(row):
        text = row["text"].lower()
        async with gate:
            ref = await call_tool(api, stage="bench2_ref", system=REF_SYSTEM, tool=REF_TOOL,
                                  user=json.dumps({"transmission": text}))
            history = []
            for _ in range(2):
                check = await call_tool(api, stage="bench2_verify", system=VERIFY_SYSTEM, tool=VERIFY_TOOL,
                                        user=json.dumps({"transmission": text, "reference": ref}))
                history.append(check["problems"])
                if check["correct"]:
                    break
                ref = check["corrected"]
            else:
                check = {"correct": False}
        with path.open("a") as out:
            out.write(json.dumps({"id": row["id"], "component": row["component"], "text": text, **ref,
                                  "ref_config": REF_CONFIG_HASH,
                                  "verified": check["correct"], "revisions": len(history) - check["correct"],
                                  "verify_history": history}) + "\n")

    await asyncio.gather(*(run(r) for r in rows_for(args.split) if r["id"] not in done))
    refs = jsonl(path)
    print(f"references: {len(refs)}; disputed (excluded): {sum(not r['verified'] for r in refs)}")
    (BENCH / f"references-{args.split}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")


async def resolve(args):
    """Up to two more independent verification rounds for disputed references (before any scoring)."""
    path = BENCH / f"references-{args.split}.jsonl"
    rows = jsonl(path)
    if any(r.get("ref_config") != REF_CONFIG_HASH for r in rows):
        raise SystemExit(f"{path.name} has rows from a different reference configuration; not resolving")
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(row):
        ref = {k: row[k] for k in REFERENCE_SCHEMA["required"]}
        async with gate:
            for _ in range(2):
                check = await call_tool(api, stage="bench2_resolve", system=VERIFY_SYSTEM, tool=VERIFY_TOOL,
                                        user=json.dumps({"transmission": row["text"], "reference": ref}))
                row["verify_history"].append(check["problems"])
                if check["correct"]:
                    row.update(ref, verified=True, resolved=True)
                    return
                ref = check["corrected"]
        row["revisions"] = row.get("revisions", 0) + 2

    await asyncio.gather(*(run(r) for r in rows if not r["verified"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (BENCH / f"references-{args.split}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    print(f"resolved {sum(bool(r.get('resolved')) for r in rows)}; still disputed {sum(not r['verified'] for r in rows)}")


async def run_arm(args):
    BENCH.mkdir(parents=True, exist_ok=True)
    system, prompt_name = args.arm.split(":")
    path = BENCH / f"outputs-{args.split}-{system}-{prompt_name}.jsonl"
    prompt_hash = hashlib.sha256(PROMPTS[prompt_name].encode()).hexdigest()[:16]
    if args.kind == "nemotron":
        if not (args.url and args.checkpoint):
            raise SystemExit("nemotron arms need --url and --checkpoint")
        api = AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", timeout=600,
                          api_key=dotenv_values(args.env_file)["GB_SERVING_API_KEY"])
        params = {"model": "gb-phone", **DECODING, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    else:
        api = AsyncOpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"], timeout=600)
        params = {"model": args.openai_model, "reasoning_effort": "medium", "max_completion_tokens": 4096}
    identity = {"arm": args.arm, "kind": args.kind, "url": args.url, "checkpoint": args.checkpoint,
                "prompt_hash": prompt_hash, "params": params,
                "split_sha256": hashlib.sha256((OUT / "split.jsonl").read_bytes()).hexdigest()[:16]}
    existing = jsonl(path)
    if any(r["identity"] != identity for r in existing):
        raise SystemExit(f"{path.name} has rows from a different arm identity; use a new arm name")
    done = {r["id"] for r in existing}
    gate = asyncio.Semaphore(args.concurrency)

    async def run(row):
        messages = [{"role": "system", "content": PROMPTS[prompt_name]}, {"role": "user", "content": row["text"].lower()}]
        async with gate:
            try:
                response = await api.chat.completions.create(messages=messages, **params)
                choice = response.choices[0]
                record = {"answer": (choice.message.content or "").strip(), "error": None,
                          "served_model": response.model, "finish_reason": choice.finish_reason}
            except Exception as exc:  # noqa: BLE001 - recorded and scored as incorrect
                record = {"answer": "", "error": f"{type(exc).__name__}: {exc}"[:300],
                          "served_model": None, "finish_reason": None}
        with path.open("a") as out:
            out.write(json.dumps({"id": row["id"], "identity": identity, **record}) + "\n")

    await asyncio.gather(*(run(r) for r in rows_for(args.split) if r["id"] not in done))
    print(path.name, len(jsonl(path)))


JUDGE_CONFIG_HASH = hashlib.sha256(json.dumps([JUDGE_SYSTEM, JUDGE_TOOL, "claude-opus-5-5", "high"]).encode()).hexdigest()[:16]


def judge_key(output, ref):
    """Hash of every semantic judge input: transmission, reference, answer, error state, judge config."""
    payload = json.dumps([ref["text"], {k: ref[k] for k in REFERENCE_SCHEMA["required"]},
                          output["answer"], bool(output["error"]), JUDGE_CONFIG_HASH])
    return hashlib.sha256(payload.encode()).hexdigest()


async def judge(args):
    refs = {r["id"]: r for r in jsonl(BENCH / f"references-{args.split}.jsonl")}
    api, gate = client(), asyncio.Semaphore(args.concurrency)
    for path in sorted(BENCH.glob(f"outputs-{args.split}-*.jsonl")):
        judged_path = path.with_name(path.name.replace("outputs-", "judged-"))
        cached = {r["key"]: r["judge"] for r in jsonl(judged_path)}
        results = []

        async def run(output):
            ref = refs[output["id"]]
            key = judge_key(output, ref)
            if key in cached:
                results.append({"id": output["id"], "key": key, "judge_hash": JUDGE_CONFIG_HASH,
                                "answer": output["answer"], "error": output["error"],
                                "identity": output["identity"], "judge": cached[key]})
                return
            if output["error"] or not output["answer"]:
                verdict = {"missing_required": ["(no answer)"], "unsupported": [], "speaker_acceptable": False,
                           "one_sentence": False, "plain_spoken": False}
            else:
                async with gate:
                    verdict = await call_tool(api, stage="bench2_judge", system=JUDGE_SYSTEM, tool=JUDGE_TOOL, user=json.dumps({
                        "transmission": ref["text"],
                        "reference": {k: ref[k] for k in REFERENCE_SCHEMA["required"]},
                        "answer": output["answer"]}))
            results.append({"id": output["id"], "key": key, "judge_hash": JUDGE_CONFIG_HASH, "answer": output["answer"],
                            "error": output["error"], "identity": output["identity"], "judge": verdict})

        await asyncio.gather(*(run(o) for o in jsonl(path)))
        judged_path.write_text("".join(json.dumps(r) + "\n" for r in sorted(results, key=lambda r: r["id"])))
        print(judged_path.name, len(results))
    print("ledger:", json.dumps(ledger_summary()))


def correct(record):
    j = record["judge"]
    return not j["missing_required"] and not j["unsupported"] and j["speaker_acceptable"]


def report(args):
    ref_rows = jsonl(BENCH / f"references-{args.split}.jsonl")
    refs = {r["id"]: r for r in ref_rows}
    cohort = {r["id"] for r in rows_for(args.split)}
    if len(ref_rows) != len(refs) or set(refs) != cohort:
        raise SystemExit(f"references incomplete or duplicated: {len(refs)} unique of {len(cohort)} frozen rows")
    scored_ids = sorted(i for i, r in refs.items() if r["verified"])
    if not scored_ids:
        raise SystemExit("no verified references: report unavailable")
    arms = {}
    for path in sorted(BENCH.glob(f"judged-{args.split}-*.jsonl")):
        records = jsonl(path)
        rows = {r["id"]: r for r in records}
        arm = path.stem.replace(f"judged-{args.split}-", "")
        outputs = {o["id"]: o for o in jsonl(path.with_name(path.name.replace("judged-", "outputs-")))}
        problems = []
        if set(outputs) != cohort:
            problems.append(f"outputs file covers {len(set(outputs) & cohort)} of {len(cohort)} cohort rows")
        if len(records) != len(rows):
            problems.append("duplicate ids")
        if set(rows) != cohort:
            problems.append(f"{len(cohort - set(rows))} cohort rows missing")
        if len({json.dumps(r["identity"], sort_keys=True) for r in records}) > 1:
            problems.append("mixed identities")
        stale = sum(1 for i, r in rows.items() if i in outputs and r["key"] != judge_key(outputs[i], refs[i]))
        if stale:
            problems.append(f"{stale} stale judgments")
        if problems:
            print(f"skip {arm}: {'; '.join(problems)}")
            continue
        arms[arm.replace("-short", ":short").replace("-full", ":full")] = [rows[i] for i in scored_ids]
    groups = defaultdict(list)
    for index, row_id in enumerate(scored_ids):
        groups[refs[row_id]["component"]].append(index)
    group_list = list(groups.values())
    rng = random.Random(20261009)
    draws = [[i for g in (rng.choice(group_list) for _ in group_list) for i in g] for _ in range(2000)]

    def rate(rows, pick, indices=None):
        indices = range(len(rows)) if indices is None else indices
        return sum(bool(pick(rows[i])) for i in indices) / max(len(indices), 1)

    def interval(values):
        values = sorted(values)
        return values[49], values[1949]

    baseline = arms.get(args.baseline)
    disputed = len(refs) - len(scored_ids)
    lines = [f"Rubric v2, split `{args.split}`: {len(scored_ids)} of {len(refs)} frozen rows scored "
             f"({disputed} disputed references excluded; full-cohort correctness lies between the scored rate "
             f"× {len(scored_ids)}/{len(refs)} and that plus {disputed / len(refs):.1%}). Baseline `{args.baseline}`.", "",
             "| Arm | Correct (95% CI) | Δ vs baseline (95% CI) | Unsupported | Missing required | Speaker ok | One sentence | Plain spoken | Errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, rows in arms.items():
        lo, hi = interval([rate(rows, correct, d) for d in draws])
        delta = ""
        if baseline:
            d = rate(rows, correct) - rate(baseline, correct)
            dlo, dhi = interval([rate(rows, correct, x) - rate(baseline, correct, x) for x in draws])
            delta = f"{d:+.1%} ({dlo:+.1%}–{dhi:+.1%})"
        lines.append(
            f"| {arm} | {rate(rows, correct):.1%} ({lo:.1%}–{hi:.1%}) | {delta} | "
            f"{rate(rows, lambda r: r['judge']['unsupported']):.1%} | "
            f"{rate(rows, lambda r: r['judge']['missing_required']):.1%} | "
            f"{rate(rows, lambda r: r['judge']['speaker_acceptable']):.1%} | "
            f"{rate(rows, lambda r: r['judge']['one_sentence']):.1%} | "
            f"{rate(rows, lambda r: r['judge']['plain_spoken']):.1%} | "
            f"{rate(rows, lambda r: r['error'] or not r['answer']):.1%} |")
    if baseline and args.candidate in arms:
        cand = arms[args.candidate]
        gain = rate(cand, correct) - rate(baseline, correct)
        unsupported_ok = rate(cand, lambda r: r["judge"]["unsupported"]) <= rate(baseline, lambda r: r["judge"]["unsupported"])
        unsup = lambda r: r["judge"]["unsupported"]
        ulo, uhi = interval([rate(cand, unsup, x) - rate(baseline, unsup, x) for x in draws])
        verdict = "SUCCESS (point rule)" if gain >= 0.05 and unsupported_ok else "NOT MET (point rule)"
        scope = "final test" if args.split == "test" else "selection/development diagnostic, not a final result"
        lines += ["", f"Success rule, {scope} ({args.candidate} vs {args.baseline}): gain {gain:+.1%} (needs ≥ +5.0); "
                      f"unsupported-claim difference {rate(cand, unsup) - rate(baseline, unsup):+.1%} "
                      f"(95% CI {ulo:+.1%}–{uhi:+.1%}; must be ≤ 0) → **{verdict}**"]
    text = "\n".join(lines)
    (BENCH / f"report-{args.split}.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["references", "resolve", "run", "judge", "report"])
    parser.add_argument("--split", choices=["selection", "test", "devdiag", "devpool"], required=True)
    parser.add_argument("--arm", help="system:prompt, prompt in short|full")
    parser.add_argument("--kind", choices=["nemotron", "openai"], default="nemotron")
    parser.add_argument("--url")
    parser.add_argument("--checkpoint", help="served model path, recorded in the arm identity")
    parser.add_argument("--env-file", default=str(ROOT.parent / "zorana-research-internal/.env"))
    parser.add_argument("--openai-model", default="gpt-5.6-terra")
    parser.add_argument("--baseline", default="stock:short", help="dev-selected stock arm for deltas")
    parser.add_argument("--candidate", help="arm evaluated against the success rule")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.command == "references":
        asyncio.run(references(args))
    elif args.command == "resolve":
        asyncio.run(resolve(args))
    elif args.command == "run":
        asyncio.run(run_arm(args))
    elif args.command == "judge":
        asyncio.run(judge(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
