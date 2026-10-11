"""v1.3 benchmark: references, runs, judge, mode suite, judge challenge set and report.

Cohorts (artifacts/atc-tutor/v13/split.jsonl, `part13`):
  dev      devdiag + selection rows: checkpoint development
  reserve  newly withheld devpool components: references only, scored once for promotion
           (`run`/`judge` refuse it unless --promotion is given)
  mode     the frozen mode/chat suite (bench/v13/mode-suite.jsonl)
The consumed final test (`part13 == "test"`) is never referenced, run or judged; its text
is read only by the mode-suite overlap filter.

references: candidate-blind v1.3 references (v2 fields plus ambiguity, licensed readings
            and prohibited completions), then a separate verify call that may correct
            (max 2 revisions). Still disputed -> excluded from scoring.
run:        one completion per row; arm = <system>:<prompt>, prompt in short13|full13|short|full.
judge:      blinded v1.3 judge with a severe-error rate; cached by a hash of every input.
challenge:  correct answers and deliberate near-misses on TRAIN transcripts, judged to
            measure false accepts of severe errors and false rejects of correct answers.
report:     strict correctness, severe-error rate and types, paired component bootstrap
            vs --baseline, and the mode-suite table.

Usage:
  bench13.py references --cohort dev
  bench13.py mode-suite
  bench13.py challenge
  bench13.py run --cohort dev --arm stock:full13 --url URL --checkpoint PATH
  bench13.py run --cohort dev --arm terra:full13 --kind openai
  bench13.py judge --cohort dev
  bench13.py report --baseline v2i328:short
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict

from dotenv import dotenv_values
from openai import AsyncOpenAI

import split13
import telephony
from opus import OUT, ROOT, call_tool, client, ledger_summary

sys.path.insert(0, str(ROOT))
from tutor_prompts import FULL13_PROMPT, FULL_PROMPT, SHORT13_PROMPT, TUTOR_PROMPT  # noqa: E402

BENCH = OUT / "bench" / "v13"
CONTRACT = (ROOT / "docs/atc-tutor-contract-v13.md").read_text()
TELEPHONY = json.loads((OUT / "telephony-v13.json").read_text())["entries"]
TELEPHONY_TEXT = "; ".join(f"{k} = {v['airline']}" for k, v in sorted(TELEPHONY.items()))
PROMPTS = {"short13": SHORT13_PROMPT, "full13": FULL13_PROMPT, "short": TUTOR_PROMPT, "full": FULL_PROMPT}
DECODING = {"temperature": 0, "max_tokens": 200}
SEVERE_TYPES = ["wrong_value", "instruction_changed", "lost_negation_condition", "omitted_action",
                "wrong_callsign", "role_mode_error", "unlicensed_probably"]

REFERENCE_SCHEMA = {
    "type": "object",
    "required": ["speakers", "required", "optional", "alternative_readings", "acceptable_uncertainty",
                 "ambiguity", "ambiguity_evidence", "licensed_readings", "prohibited_completions"],
    "properties": {
        "speakers": {"type": "array", "items": {"enum": ["controller", "pilot", "speaker"]},
                     "description": "Every acceptable attribution. Include 'speaker' whenever the role is not fixed by a real cue; a trailing callsign is never a cue."},
        "required": {"type": "array", "items": {"type": "string"},
                     "description": "Material operational propositions: speech act, every operational value and its meaning, conditions, negation, clearance vs expectation, corrections, and for questions that it is a question."},
        "optional": {"type": "array", "items": {"type": "string"},
                     "description": "Identity and courtesy items an answer may include but need not: callsign as heard, airline gloss from the table, station name, greetings."},
        "alternative_readings": {"type": "array", "items": {"type": "string"}},
        "acceptable_uncertainty": {"type": "string"},
        "ambiguity": {"type": "string", "enum": ["clear", "plausible_reading", "unresolved"]},
        "ambiguity_evidence": {"type": "string"},
        "licensed_readings": {"type": "array", "items": {"type": "string"},
                              "description": "Readings of garbled values an answer may give with 'probably'. Empty unless ambiguity is plausible_reading."},
        "prohibited_completions": {"type": "array", "items": {"type": "string"},
                                   "description": "Tempting but unsupported completions or readings that count as severe errors."},
    },
}
REF_FIELDS = REFERENCE_SCHEMA["required"]
REF_TOOL = {"name": "reference", "input_schema": REFERENCE_SCHEMA}
VERIFY_TOOL = {"name": "verify", "input_schema": {
    "type": "object", "required": ["correct", "problems", "corrected"],
    "properties": {"correct": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}},
                   "corrected": REFERENCE_SCHEMA}}}
JUDGE_TOOL = {"name": "judge", "input_schema": {
    "type": "object",
    "required": ["missing_required", "unsupported", "speaker_ok", "mode_ok", "severe", "severe_types",
                 "severe_notes", "one_sentence", "plain_spoken"],
    "properties": {
        "missing_required": {"type": "array", "items": {"type": "string"}},
        "unsupported": {"type": "array", "items": {"type": "string"},
                        "description": "Operational claims not supported by the transmission and the reference: invented or changed values, wrong clearance type, invented purpose or context, unlicensed readings, wrong or invented callsign letters or airline."},
        "speaker_ok": {"type": "boolean"},
        "mode_ok": {"type": "boolean", "description": "False if the answer replies to the transmission as if addressed to the tutor instead of explaining it."},
        "severe": {"type": "boolean"},
        "severe_types": {"type": "array", "items": {"type": "string", "enum": SEVERE_TYPES}},
        "severe_notes": {"type": "string"},
        "one_sentence": {"type": "boolean"},
        "plain_spoken": {"type": "boolean", "description": "Plain language, numbers in words, no quoting, readback or markdown."},
    }}}

RUBRIC = """Rubric v3 (contract v1.3): an answer is correct only if it conveys every REQUIRED
proposition (any wording; a listed alternative reading also satisfies it), makes no
unsupported operational claim, uses an acceptable speaker, and explains rather than
answers (mode). OPTIONAL items are never required. For ambiguity `unresolved`, a concise
statement that the value or part is unclear satisfies the affected proposition. For
`plausible_reading`, either a licensed reading marked "probably" or a concise uncertainty
statement satisfies it; any reading not licensed, or a licensed reading stated as certain
when the reference marks it uncertain, is unsupported. Anything in prohibited_completions
is unsupported and severe. Callsigns: the callsign as heard is always fine; dropped,
added or changed letters, an unspoken nationality prefix, or an airline not given by the
telephony table is a wrong_callsign. Style (length, verb choice, gloss wording) never makes
an answer incorrect; judge meaning.
Severe errors, per the contract: wrong_value (wrong operational value or unit),
instruction_changed (instruction, permission or clearance reversed or changed in kind),
lost_negation_condition, omitted_action (an omitted action that changes what the aircraft
would do), wrong_callsign (wrong or invented callsign, recipient or airline),
role_mode_error (answering the transmission, or the wrong speaker when the text fixes it),
unlicensed_probably. Other failures are minor."""

COMMON = f"""Contract v1.3:

{CONTRACT}

{RUBRIC}

Checked airline telephony table (only these telephony words may be glossed with an
airline): {TELEPHONY_TEXT}"""

REF_SYSTEM = f"""You write evaluation references for an ATC phrase tutor. {COMMON}

From the transmission alone, write the reference: acceptable speakers, required
operational propositions, optional identity/courtesy items, alternative faithful
readings, acceptable uncertainty, the ambiguity status with its evidence, readings
licensed for "probably", and tempting completions that would be severe errors. Never
license a reading of cut-off digits or a value with two plausible readings."""
VERIFY_SYSTEM = f"""You independently check an evaluation reference for an ATC phrase tutor
against the transmission. {COMMON}

Mark correct only if every field is right and internally consistent: required items are
material and supported; identity items are optional; every faithful reading and speaker
is accepted ('speaker' always accepted when no real cue fixes the role; a trailing
callsign is not a cue); the ambiguity status is right; licensed readings are clearly most
plausible from the visible words and the phrase's role, never guesses; prohibited
completions are genuinely unsupported. Otherwise return the full corrected reference."""
JUDGE_SYSTEM = f"""You grade one answer from an ATC phrase tutor. You do not know which
system wrote it. {COMMON}

Compare the answer with the transmission and the reference. List missing REQUIRED
propositions and unsupported operational claims; say whether the speaker is acceptable
and whether the answer explains (mode); decide whether any error is severe and which
severe types apply."""

REF_CONFIG_HASH = hashlib.sha256(json.dumps(
    [REF_SYSTEM, VERIFY_SYSTEM, REF_TOOL, VERIFY_TOOL, "claude-opus-5-5", "high"]).encode()).hexdigest()[:16]
JUDGE_CONFIG_HASH = hashlib.sha256(json.dumps(
    [JUDGE_SYSTEM, JUDGE_TOOL, "claude-opus-5-5", "high"]).encode()).hexdigest()[:16]

# ---------------------------------------------------------------- mode suite
MODE_KINDS = {
    "explain_question": "an ATC transmission phrased as a question (controller or pilot asking something on frequency); the tutor must EXPLAIN it, never answer it",
    "tutor_question": "the user asks the tutor an aviation question directly (what a term means, why something is done); the tutor must ANSWER briefly and correctly",
    "contrast_pair": "two items in one family: the same idea once as an on-frequency question to EXPLAIN and once as a user question to the tutor to ANSWER",
    "numbers_terms": "a user question about an aviation number or term (flight levels, QNH, squawk codes, headings, FATO, CTR); ANSWER with exact facts",
    "chat": "ordinary conversation with the tutor (greetings, thanks, what can you do, off-topic small talk); CHAT briefly",
    "multi_turn": "a short history (user transmission, assistant explanation) then a follow-up: 'repeat that', 'what does that mean for the pilot', 'and the second part?', or a new transmission; the tutor must RESTATE, ANSWER or EXPLAIN as appropriate",
}
MODE_ITEM = {"type": "object", "required": ["family", "kind", "messages", "expected_mode", "must"],
             "properties": {
                 "family": {"type": "string"},
                 "kind": {"type": "string", "enum": list(MODE_KINDS)},
                 "messages": {"type": "array", "items": {
                     "type": "object", "required": ["role", "content"],
                     "properties": {"role": {"type": "string", "enum": ["user", "assistant"]}, "content": {"type": "string"}}}},
                 "expected_mode": {"type": "string", "enum": ["explain", "answer", "chat", "restate"]},
                 "must": {"type": "string", "description": "What a correct reply must do and the exact facts it must get right."},
             }}
MODE_GEN_TOOL = {"name": "items", "input_schema": {
    "type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": MODE_ITEM}}}}
MODE_GEN_SYSTEM = f"""You write a frozen test suite for an ATC phrase tutor's conversation
behavior. {COMMON}

Write NEW, realistic, varied items. User transmissions are lower-case speech-to-text,
like real ATC radio (callsigns, numbers as words, occasional disfluency). Never copy a
real recording. Vary airports, callsigns, airlines (from the table), values and phrasing.
Each item's messages end with a user turn. Give each item a short unique family id
(prefix with the kind); both members of a contrast pair share one family id."""
MODE_JUDGE_TOOL = {"name": "mode_judge", "input_schema": {
    "type": "object", "required": ["mode_correct", "factual_error", "factual_notes", "style_ok"],
    "properties": {"mode_correct": {"type": "boolean", "description": "The reply explains, answers, chats or restates as expected."},
                   "factual_error": {"type": "boolean", "description": "Any wrong fact, value, unit or invented detail."},
                   "factual_notes": {"type": "string"},
                   "style_ok": {"type": "boolean", "description": "Brief spoken style: one or two short sentences, numbers in words, no markdown or lists."}}}}
MODE_JUDGE_SYSTEM = f"""You grade one reply from an ATC phrase tutor in a conversation. {COMMON}

You get the conversation, the expected mode and what a correct reply must do."""
MODE_JUDGE_HASH = hashlib.sha256(json.dumps([MODE_JUDGE_SYSTEM, MODE_JUDGE_TOOL, "claude-opus-5-5", "high"]).encode()).hexdigest()[:16]

# ---------------------------------------------------------------- challenge set
CORRUPTIONS = {
    "changed_digit": "change exactly one digit of one operational value",
    "negation_dropped": "drop or reverse a negation, condition or restriction (or, if none, turn a permission into an order)",
    "wrong_callsign": "change, add or drop one callsign letter or digit, or add an unspoken nationality prefix",
    "invented_purpose": "add a plausible but unstated reason, purpose, destination or approach type",
    "unlicensed_probably": "add a 'probably' reading of a value or word that the transmission does not license",
    "answered": "reply to the transmission as if it were addressed to the tutor instead of explaining it",
    "trailing_pilot": "attribute the transmission to the pilot as a readback just because the callsign comes last (only where the role is not fixed)",
}
CHALLENGE_TOOL = {"name": "challenge", "input_schema": {
    "type": "object", "required": ["correct_answer", "near_misses"],
    "properties": {
        "correct_answer": {"type": "string", "description": "A fully correct v1.3 answer in the house style."},
        "near_misses": {"type": "array", "items": {
            "type": "object", "required": ["corruption", "answer"],
            "properties": {"corruption": {"type": "string", "enum": list(CORRUPTIONS)}, "answer": {"type": "string"}}}},
    }}}
CHALLENGE_SYSTEM = f"""You build a judge-calibration set for an ATC phrase tutor. {COMMON}

Given a transmission and its reference, write one fully correct answer and the requested
near-miss answers. Each near miss changes exactly one thing, as described, and otherwise
stays as close as possible to the correct answer. Corruption types:
""" + "\n".join(f"- {k}: {v}" for k, v in CORRUPTIONS.items())


def jsonl(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_rows(cohort):
    parts = {"dev": {"devdiag", "selection"}, "reserve": {"reserve"}}[cohort]
    return [r for r in split13.load().values() if r["part13"] in parts]


def ref_path(cohort):
    return BENCH / f"references-{cohort}.jsonl"


async def build_reference(api, text, stage):
    ref = await call_tool(api, stage=f"{stage}_ref", system=REF_SYSTEM, tool=REF_TOOL,
                          user=json.dumps({"transmission": text}))
    history = []
    for _ in range(3):
        check = await call_tool(api, stage=f"{stage}_verify", system=VERIFY_SYSTEM, tool=VERIFY_TOOL,
                                user=json.dumps({"transmission": text, "reference": ref}))
        history.append(check["problems"])
        if check["correct"]:
            return ref, True, history
        ref = check["corrected"]
    return ref, False, history


async def references(args):
    BENCH.mkdir(parents=True, exist_ok=True)
    path = ref_path(args.cohort)
    existing = jsonl(path)
    if any(r.get("ref_config") != REF_CONFIG_HASH for r in existing):
        raise SystemExit(f"{path.name} was written with a different reference configuration")
    done = {r["id"] for r in existing}
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(row):
        text = row["text"].lower()
        async with gate:
            ref, verified, history = await build_reference(api, text, "bench13")
        with path.open("a") as out:
            out.write(json.dumps({"id": row["id"], "component": row["component"], "part13": row["part13"],
                                  "text": text, **{k: ref[k] for k in REF_FIELDS}, "ref_config": REF_CONFIG_HASH,
                                  "verified": verified, "revisions": len(history) - verified,
                                  "verify_history": history}) + "\n")

    await asyncio.gather(*(run(r) for r in cohort_rows(args.cohort) if r["id"] not in done))
    refs = jsonl(path)
    (BENCH / f"references-{args.cohort}.sha256").write_text(sha(path) + "\n")
    print(f"{path.name}: {len(refs)}; disputed (excluded): {sum(not r['verified'] for r in refs)}; "
          f"ambiguity {dict(Counter(r['ambiguity'] for r in refs))}")


def normalize(text):
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def held_out_ngrams(n=6):
    """6-grams of every non-train transmission (dev, reserve and the consumed test)."""
    grams = set()
    for r in split13.load().values():
        if r["part13"] != "train":
            words = normalize(r["text"])
            grams.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
            if 0 < len(words) < n:
                grams.add(tuple(words))
    return grams


async def mode_suite(args):
    path = BENCH / "mode-suite.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; the suite is frozen")
    BENCH.mkdir(parents=True, exist_ok=True)
    api = client()
    plan = [("explain_question", 24), ("tutor_question", 16), ("contrast_pair", 24),
            ("numbers_terms", 18), ("chat", 16), ("multi_turn", 26)]

    async def gen(kind, n):
        result = await call_tool(api, stage="bench13_mode_gen", system=MODE_GEN_SYSTEM, tool=MODE_GEN_TOOL,
                                 user=json.dumps({"kind": kind, "description": MODE_KINDS[kind], "count": n,
                                                  "note": "contrast_pair: write count/2 families of two items each"}))
        return [dict(i, kind=kind) for i in result["items"]]

    batches = await asyncio.gather(*(gen(k, n) for k, n in plan))
    grams = held_out_ngrams()
    items, dropped = [], 0
    for batch in batches:
        for item in batch:
            user_texts = [m["content"] for m in item["messages"] if m["role"] == "user"]
            overlap = False
            for t in user_texts:
                words = normalize(t)
                if any(tuple(words[i:i + 6]) in grams for i in range(len(words) - 5)) or tuple(words) in grams:
                    overlap = True
            if overlap or not item["messages"] or item["messages"][-1]["role"] != "user":
                dropped += 1
                continue
            items.append(item)
    families = defaultdict(list)
    for item in items:
        families[item["family"]].append(item)
    with path.open("w") as f:
        for n, item in enumerate(items, 1):
            f.write(json.dumps({"id": f"m{n:03d}", **item}) + "\n")
    (BENCH / "mode-suite-families.txt").write_text("\n".join(sorted(families)) + "\n")
    (BENCH / "mode-suite.sha256").write_text(sha(path) + "\n")
    print(path, len(items), "items,", len(families), "families; dropped", dropped,
          dict(Counter(i["kind"] for i in items)))


def arm_path(cohort, arm):
    system, prompt = arm.split(":")
    return BENCH / f"outputs-{cohort}-{system}-{prompt}.jsonl"


async def run_arm(args):
    if args.cohort == "reserve" and not args.promotion:
        raise SystemExit("the reserve cohort is scored once, for promotion only (--promotion)")
    BENCH.mkdir(parents=True, exist_ok=True)
    system, prompt_name = args.arm.split(":")
    path = arm_path(args.cohort, args.arm)
    prompt = PROMPTS[prompt_name]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if args.kind == "nemotron":
        if not (args.url and args.checkpoint):
            raise SystemExit("nemotron arms need --url and --checkpoint")
        api = AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", timeout=900,
                          api_key=dotenv_values(args.env_file)["GB_SERVING_API_KEY"])
        params = {"model": "gb-phone", **DECODING, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        if args.temperature:  # sampling diagnostics only (best-of-k); scored arms use temperature 0
            params["temperature"] = args.temperature
    else:
        api = AsyncOpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"], timeout=600)
        params = {"model": args.openai_model, "reasoning_effort": "medium", "max_completion_tokens": 4096}
    source = BENCH / "mode-suite.jsonl" if args.cohort == "mode" else split13.SPLIT
    identity = {"arm": args.arm, "kind": args.kind, "url": args.url, "checkpoint": args.checkpoint,
                "prompt_hash": prompt_hash, "params": params, "source_sha256": sha(source)[:16]}
    existing = jsonl(path)
    if any(r["identity"] != identity for r in existing):
        raise SystemExit(f"{path.name} has rows from a different arm identity; use a new arm name")
    done = {r["id"] for r in existing}
    if args.cohort == "mode":
        rows = [{"id": i["id"], "messages": i["messages"]} for i in jsonl(source)]
    else:
        rows = [{"id": r["id"], "messages": [{"role": "user", "content": r["text"].lower()}]} for r in cohort_rows(args.cohort)]
    gate = asyncio.Semaphore(args.concurrency)

    async def run(row):
        messages = [{"role": "system", "content": prompt}, *row["messages"]]
        async with gate:
            for attempt in range(3):
                try:
                    response = await api.chat.completions.create(messages=messages, **params)
                    choice = response.choices[0]
                    record = {"answer": (choice.message.content or "").strip(), "error": None,
                              "served_model": response.model, "finish_reason": choice.finish_reason}
                    break
                except Exception as exc:  # noqa: BLE001 - infrastructure retry, then recorded as incorrect
                    record = {"answer": "", "error": f"{type(exc).__name__}: {exc}"[:300],
                              "served_model": None, "finish_reason": None}
                    await asyncio.sleep(10 * (attempt + 1))
        with path.open("a") as out:
            out.write(json.dumps({"id": row["id"], "identity": identity, **record}) + "\n")

    await asyncio.gather(*(run(r) for r in rows if r["id"] not in done))
    out = jsonl(path)
    print(path.name, len(out), "errors:", sum(bool(r["error"]) for r in out))


def arm_files(args, cohort):
    """Outputs to judge: every arm, or only --arms (system:prompt, comma-separated)."""
    if not args.arms:
        return sorted(BENCH.glob(f"outputs-{cohort}-*.jsonl"))
    return [arm_path(cohort, arm) for arm in args.arms.split(",")]


def judge_key(text, ref, answer, error):
    payload = json.dumps([text, {k: ref[k] for k in REF_FIELDS}, answer, bool(error), JUDGE_CONFIG_HASH])
    return hashlib.sha256(payload.encode()).hexdigest()


async def judge_one(api, text, ref, answer, stage="bench13_judge"):
    return await call_tool(api, stage=stage, system=JUDGE_SYSTEM, tool=JUDGE_TOOL, user=json.dumps({
        "transmission": text, "reference": {k: ref[k] for k in REF_FIELDS}, "answer": answer}))


NO_ANSWER = {"missing_required": ["(no answer)"], "unsupported": [], "speaker_ok": False, "mode_ok": False,
             "severe": True, "severe_types": ["omitted_action"], "severe_notes": "no answer",
             "one_sentence": False, "plain_spoken": False}


async def judge(args):
    if args.cohort == "reserve" and not args.promotion:
        raise SystemExit("the reserve cohort is scored once, for promotion only (--promotion)")
    api, gate = client(), asyncio.Semaphore(args.concurrency)
    if args.cohort == "mode":
        return await judge_mode(args, api, gate)
    refs = {r["id"]: r for r in jsonl(ref_path(args.cohort))}
    for path in arm_files(args, args.cohort):
        judged_path = path.with_name(path.name.replace("outputs-", "judged-"))
        cached = {r["key"]: r["judge"] for r in jsonl(judged_path)}
        results = []

        async def run(output):
            ref = refs[output["id"]]
            key = judge_key(ref["text"], ref, output["answer"], output["error"])
            if key in cached:
                verdict = cached[key]
            elif output["error"] or not output["answer"]:
                verdict = NO_ANSWER
            else:
                async with gate:
                    verdict = await judge_one(api, ref["text"], ref, output["answer"])
            results.append({"id": output["id"], "key": key, "judge_hash": JUDGE_CONFIG_HASH,
                            "answer": output["answer"], "error": output["error"],
                            "identity": output["identity"], "judge": verdict})

        await asyncio.gather(*(run(o) for o in jsonl(path) if o["id"] in refs))
        judged_path.write_text("".join(json.dumps(r) + "\n" for r in sorted(results, key=lambda r: r["id"])))
        print(judged_path.name, len(results))
    print("ledger:", json.dumps({k: v for k, v in ledger_summary().items() if k.startswith("bench13")}))


async def judge_mode(args, api, gate):
    items = {i["id"]: i for i in jsonl(BENCH / "mode-suite.jsonl")}
    for path in arm_files(args, "mode"):
        judged_path = path.with_name(path.name.replace("outputs-", "judged-"))
        cached = {r["key"]: r["judge"] for r in jsonl(judged_path)}
        results = []

        async def run(output):
            item = items[output["id"]]
            key = hashlib.sha256(json.dumps([item["messages"], item["must"], output["answer"], MODE_JUDGE_HASH]).encode()).hexdigest()
            if key in cached:
                verdict = cached[key]
            elif output["error"] or not output["answer"]:
                verdict = {"mode_correct": False, "factual_error": True, "factual_notes": "no answer", "style_ok": False}
            else:
                async with gate:
                    verdict = await call_tool(api, stage="bench13_mode_judge", system=MODE_JUDGE_SYSTEM, tool=MODE_JUDGE_TOOL,
                                              user=json.dumps({"conversation": item["messages"], "expected_mode": item["expected_mode"],
                                                               "must": item["must"], "reply": output["answer"]}))
            results.append({"id": output["id"], "key": key, "answer": output["answer"], "identity": output["identity"], "judge": verdict})

        await asyncio.gather(*(run(o) for o in jsonl(path)))
        judged_path.write_text("".join(json.dumps(r) + "\n" for r in sorted(results, key=lambda r: r["id"])))
        print(judged_path.name, len(results))


def correct(j):
    return not j["missing_required"] and not j["unsupported"] and j["speaker_ok"] and j["mode_ok"]


async def challenge(args):
    """Judge calibration on TRAIN transcripts: correct answers and one-change near misses."""
    path = BENCH / "challenge.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists")
    rows = sorted((r for r in split13.load().values() if r["part13"] == "train"), key=lambda r: r["id"])
    rng = random.Random(20261010)
    long_rows = [r for r in rows if len(r["text"].split()) >= 10]
    short_rows = [r for r in rows if 4 <= len(r["text"].split()) < 10]
    picked = rng.sample(long_rows, args.size * 2 // 3) + rng.sample(short_rows, args.size - args.size * 2 // 3)
    kinds = list(CORRUPTIONS)
    api, gate = client(), asyncio.Semaphore(args.concurrency)
    records = []

    async def run(n, row):
        text = row["text"].lower()
        wanted = [kinds[(2 * n) % len(kinds)], kinds[(2 * n + 1) % len(kinds)]]
        async with gate:
            ref, verified, _ = await build_reference(api, text, "bench13_challenge")
            if not verified:
                return
            built = await call_tool(api, stage="bench13_challenge_gen", system=CHALLENGE_SYSTEM, tool=CHALLENGE_TOOL,
                                    user=json.dumps({"transmission": text, "reference": ref, "near_misses_wanted": wanted}))
            answers = [("correct", built["correct_answer"])] + [(m["corruption"], m["answer"]) for m in built["near_misses"]]
            for kind, answer in answers:
                verdict = await judge_one(api, text, ref, answer, stage="bench13_challenge_judge")
                records.append({"source_id": row["id"], "transmission": text, "reference": {k: ref[k] for k in REF_FIELDS},
                                "kind": kind, "answer": answer, "judge": verdict})

    await asyncio.gather(*(run(n, r) for n, r in enumerate(picked)))
    records.sort(key=lambda r: (r["source_id"], r["kind"] != "correct", r["kind"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    good = [r for r in records if r["kind"] == "correct"]
    bad = [r for r in records if r["kind"] != "correct"]
    false_reject = sum(not correct(r["judge"]) for r in good)
    false_accept = sum(correct(r["judge"]) for r in bad)
    not_severe = sum(not r["judge"]["severe"] for r in bad)
    by_kind = defaultdict(lambda: [0, 0, 0])
    for r in bad:
        by_kind[r["kind"]][0] += 1
        by_kind[r["kind"]][1] += correct(r["judge"])
        by_kind[r["kind"]][2] += not r["judge"]["severe"]
    summary = {"sources": len(good), "correct_answers": len(good), "false_reject": false_reject,
               "near_misses": len(bad), "false_accept": false_accept, "near_miss_not_flagged_severe": not_severe,
               "by_kind": {k: {"n": v[0], "accepted": v[1], "not_severe": v[2]} for k, v in sorted(by_kind.items())}}
    (BENCH / "challenge-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    # Owner calibration sample: 30 mixed items in the audit-page item format.
    sample = rng.sample(good, min(10, len(good))) + rng.sample(bad, min(20, len(bad)))
    rng.shuffle(sample)
    with (BENCH / "owner-calibration.jsonl").open("w") as f:
        for n, r in enumerate(sample, 1):
            j = r["judge"]
            f.write(json.dumps({"id": f"c{n:03d}", "n": n, "sourceId": r["source_id"], "split": "train",
                                "transmission": r["transmission"], "reference": r["reference"], "answer": r["answer"],
                                "kind": r["kind"],
                                "judge": {"missing_required": j["missing_required"], "unsupported": j["unsupported"],
                                          "speaker_acceptable": j["speaker_ok"], "mode_ok": j["mode_ok"],
                                          "severe": j["severe"], "severe_types": j["severe_types"]},
                                "judgedCorrect": correct(j), "classifier": None}) + "\n")


DIGIT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
               "eight": 8, "nine": 9, "niner": 9, "tree": 3, "fife": 5}
TAG_WORDS = {"xt", "unk", "spk", "noise"}


def _digits_after(words, i):
    out = []
    for w in words[i:i + 6]:
        if w in DIGIT_WORDS:
            out.append(str(DIGIT_WORDS[w]))
        elif w in ("hundred", "thousand") and out:
            out.append("00" if w == "hundred" else "000")
        else:
            break
    return int("".join(out)) if out else None


def recheck_flags(text):
    """Contract rules 10-11: transcription tags or implausible/unparseable values (over-flags on purpose)."""
    w, flags = text.lower().split(), []
    if TAG_WORDS & set(w) or re.search(r"\bne \w+", text):
        flags.append("tag")
    for i, x in enumerate(w):
        if x == "qnh":
            v = _digits_after(w, i + 1)
            if v is None or not (900 <= v <= 1100 or 2800 <= v <= 3100):
                flags.append(f"qnh:{v}")
        elif x == "heading" and (v := _digits_after(w, i + 1)) is not None and v > 360:
            flags.append(f"heading:{v}")
        elif x == "level" and i and w[i - 1] == "flight" and (v := _digits_after(w, i + 1)) is not None and v > 600:
            flags.append(f"fl:{v}")
        elif x == "decimal":
            pre, j = [], i - 1
            while j >= 0 and w[j] in DIGIT_WORDS and len(pre) < 3:
                pre.insert(0, str(DIGIT_WORDS[w[j]]))
                j -= 1
            if pre and not 108 <= int("".join(pre)) <= 137:
                flags.append(f"freq:{''.join(pre)}")
    return flags


async def refreeze(args):
    """Move references to the current contract: re-verify rows hit by rules 10-11, carry the rest."""
    path = ref_path(args.cohort)
    rows = jsonl(path)
    api, gate = client(), asyncio.Semaphore(args.concurrency)
    stats = Counter()

    async def run(row):
        if row["ref_config"] == REF_CONFIG_HASH:
            stats["current"] += 1
            return
        old = row["ref_config"]
        flags = ["all"] if args.recheck_all else recheck_flags(row["text"])
        if flags:
            ref = {k: row[k] for k in REF_FIELDS}
            history = []
            async with gate:
                for _ in range(3):
                    check = await call_tool(api, stage="bench13_refreeze_verify", system=VERIFY_SYSTEM, tool=VERIFY_TOOL,
                                            user=json.dumps({"transmission": row["text"], "reference": ref}))
                    history.append(check["problems"])
                    if check["correct"]:
                        break
                    ref = check["corrected"]
                else:
                    check = {"correct": False}
            row.update({k: ref[k] for k in REF_FIELDS}, verified=check["correct"])
            row["verify_history"] = row.get("verify_history", []) + history
            stats["rechecked_revised" if len(history) > 1 or not check["correct"] else "rechecked_unchanged"] += 1
            stats["rechecked_disputed"] += not check["correct"]
        else:
            stats["carried"] += 1
        row.setdefault("refreeze_history", []).append(row.pop("refreeze", None))
        row["refreeze"] = {"from_ref_config": old, "flags": flags, "date": "2026-10-10"}
        row["ref_config"] = REF_CONFIG_HASH

    await asyncio.gather(*(run(r) for r in rows))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (BENCH / f"references-{args.cohort}.sha256").write_text(sha(path) + "\n")
    print(path.name, dict(stats), "disputed:", sum(not r["verified"] for r in rows))


async def rechallenge(args):
    """Re-judge the frozen challenge items with the current judge."""
    path = BENCH / "challenge.jsonl"
    records = jsonl(path)
    api, gate = client(), asyncio.Semaphore(args.concurrency)

    async def run(r):
        async with gate:
            r["judge"] = await judge_one(api, r["transmission"], r["reference"], r["answer"], stage="bench13_challenge_judge")
        r["judge_hash"] = JUDGE_CONFIG_HASH

    await asyncio.gather(*(run(r) for r in records))
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    good = [r for r in records if r["kind"] == "correct"]
    bad = [r for r in records if r["kind"] != "correct"]
    summary = {"judge_hash": JUDGE_CONFIG_HASH, "correct_answers": len(good),
               "false_reject": sum(not correct(r["judge"]) for r in good), "near_misses": len(bad),
               "false_accept": sum(correct(r["judge"]) for r in bad),
               "near_miss_not_flagged_severe": sum(not r["judge"]["severe"] for r in bad),
               "accepted": [(r["kind"], r["answer"]) for r in bad if correct(r["judge"])]}
    (BENCH / "challenge-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- core check (deterministic)
CORE_CHECK = BENCH / "core-check.jsonl"
CORE_QUOTA = {"heading": 38, "level": 22, "squawk": 15, "expect": 12, "correction": 9, "negative": 12, "asr": 12}
ASR_MEANING = [  # (spoken pattern in the transmission, words one of which the answer must contain)
    (r"\bwil(?:ko|co)\b", ["comply"]),
    (r"\brod?ger\b", ["received", "understood", "heard", "roger"]),
    (r"\ba firm\b|\baffirmative\b", ["yes", "affirm", "correct"]),
    (r"\bsay a gain\b", ["repeat", "say again"]),
    (r"\bstand by\b", ["wait", "stand by", "standby"]),
    (r"\bintensions\b", ["intention"]),
    (r"\bdepart sure\b", ["departure", "depart"]),
]
EMERGENCY = {"seven seven zero zero": ["emergency"], "seven six zero zero": ["radio", "communication"],
             "seven five zero zero": ["hijack", "unlawful"]}


def core_norm(text):
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


def core_build(args):
    import core13
    from values import Sampler
    if CORE_CHECK.exists():
        raise SystemExit(f"{CORE_CHECK} exists; the core check is frozen")
    seen = {core_norm(r["transmission"]) for r in jsonl(OUT / "v13" / "core-inputs.jsonl")}
    rng, sampler = random.Random(args.seed), Sampler(args.seed)
    rows, skipped = [], 0
    for kind, quota in CORE_QUOTA.items():
        templates = core13.TEMPLATES[kind][1]
        n = 0
        while sum(r["kind"] == kind for r in rows) < quota:
            dialect = rng.choice(["icao_europe", "icao_europe", "icao_australia", "faa"])
            text, facts = core13.render(rng, sampler, templates[n % len(templates)], dialect)
            n += 1
            key = core_norm(text)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            rows.append({"id": f"cc-{kind}-{len(rows):03d}", "kind": kind, "transmission": text, "facts": facts})
    BENCH.mkdir(parents=True, exist_ok=True)
    CORE_CHECK.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (BENCH / "core-check.sha256").write_text(sha(CORE_CHECK) + "\n")
    trained = {core_norm(r["transmission"]) for r in jsonl(OUT / "v13" / "core-inputs.jsonl")}
    exact = {r["transmission"] for r in jsonl(OUT / "v13" / "core-inputs.jsonl")}
    print(CORE_CHECK, len(rows), "items; skipped overlaps:", skipped,
          "; exact overlaps:", sum(r["transmission"] in exact for r in rows),
          "; normalized overlaps:", sum(core_norm(r["transmission"]) in trained for r in rows))


def core_problems(item, answer):
    import data13
    from values import words_int
    text, low = item["transmission"], answer.lower().replace("-", " ")
    problems = list(data13.heading_problems(text, answer)) + list(data13.implausible(answer))
    words = text.split()
    levels = []
    for i in range(len(words) - 2):
        if words[i] == "flight" and words[i + 1] == "level":
            digits = []
            for w in words[i + 2:i + 5]:
                if w in DIGIT_WORDS:
                    digits.append(str(DIGIT_WORDS[w]))
                else:
                    break
            if digits:
                corrected = "correction" in words[i + 2 + len(digits):i + 4 + len(digits)]
                levels.append((int("".join(digits)), corrected))
    for value, superseded in levels:
        feet = words_int(value * 100).replace("-", " ")
        if not superseded and feet not in low:
            problems.append(f"flight level {value} should be about {feet} feet")
    for code, meaning in EMERGENCY.items():
        if f"squawk {code}" in text or f"squawking {code}" in text:
            if not any(m in low for m in meaning):
                problems.append(f"squawk {code} meaning missing ({'/'.join(meaning)})")
    if item["kind"] == "asr":
        for pattern, meaning in ASR_MEANING:
            if re.search(pattern, text) and not any(m in low for m in meaning):
                problems.append(f"'{re.search(pattern, text).group(0)}' meaning missing ({'/'.join(meaning)})")
    return problems


async def core(args):
    """Run an arm on the frozen core check and score it deterministically."""
    items = jsonl(CORE_CHECK)
    system, prompt_name = args.arm.split(":")
    path = BENCH / f"core-{system}-{prompt_name}.jsonl"
    prompt = PROMPTS[prompt_name]
    api = AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", timeout=900,
                      api_key=dotenv_values(args.env_file)["GB_SERVING_API_KEY"])
    params = {"model": "gb-phone", **DECODING, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    identity = {"arm": args.arm, "url": args.url, "checkpoint": args.checkpoint,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16], "params": params,
                "source_sha256": sha(CORE_CHECK)[:16]}
    existing = {r["id"]: r for r in jsonl(path) if r["identity"] == identity}
    gate = asyncio.Semaphore(args.concurrency)

    async def run(item):
        async with gate:
            for attempt in range(3):
                try:
                    response = await api.chat.completions.create(
                        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": item["transmission"]}], **params)
                    return {"id": item["id"], "identity": identity, "answer": (response.choices[0].message.content or "").strip(), "error": None}
                except Exception as exc:  # noqa: BLE001 - infrastructure retry
                    await asyncio.sleep(10 * (attempt + 1))
            return {"id": item["id"], "identity": identity, "answer": "", "error": f"{type(exc).__name__}"}

    new = await asyncio.gather(*(run(i) for i in items if i["id"] not in existing))
    rows = {**existing, **{r["id"]: r for r in new}}
    by_id = {i["id"]: i for i in items}
    for r in rows.values():
        r["problems"] = ["no answer"] if r["error"] or not r["answer"] else core_problems(by_id[r["id"]], r["answer"])
    path.write_text("".join(json.dumps(rows[i["id"]]) + "\n" for i in items))
    kinds = defaultdict(lambda: [0, 0])
    for i in items:
        kinds[i["kind"]][0] += not rows[i["id"]]["problems"]
        kinds[i["kind"]][1] += 1
    passed = sum(not r["problems"] for r in rows.values())
    print(f"{args.arm}: core check {passed}/{len(items)} ({passed / len(items):.1%}) pass;",
          ", ".join(f"{k} {p}/{n}" for k, (p, n) in kinds.items()))
    for r in [r for r in rows.values() if r["problems"]][:args.show]:
        print("  -", by_id[r["id"]]["transmission"], "->", r["answer"][:160], "|", "; ".join(r["problems"]))


def report(args):
    lines = []
    refs = {r["id"]: r for r in jsonl(ref_path("dev"))}
    scored = sorted(i for i, r in refs.items() if r["verified"])
    arms = {}
    for path in sorted(BENCH.glob("judged-dev-*.jsonl")):
        rows = {r["id"]: r for r in jsonl(path)}
        if not set(scored) <= set(rows) or any(r["judge_hash"] != JUDGE_CONFIG_HASH for r in rows.values()):
            print("skip incomplete or stale", path.name)
            continue
        name = path.stem.replace("judged-dev-", "")
        system, prompt = name.rsplit("-", 1)
        arms[f"{system}:{prompt}"] = [rows[i] for i in scored]
    groups = defaultdict(list)
    for index, row_id in enumerate(scored):
        groups[refs[row_id]["component"]].append(index)
    group_list = list(groups.values())
    rng = random.Random(20261010)
    draws = [[i for g in (rng.choice(group_list) for _ in group_list) for i in g] for _ in range(2000)]

    def rate(rows, pick, idx=None):
        idx = range(len(rows)) if idx is None else idx
        return sum(bool(pick(rows[i])) for i in idx) / max(len(idx), 1)

    def ci(values):
        values = sorted(values)
        return values[49], values[1949]

    ok = lambda r: correct(r["judge"]) and not r["error"]  # noqa: E731
    sev = lambda r: r["judge"]["severe"] or r["error"]  # noqa: E731
    base = arms.get(args.baseline)
    lines += [f"Rubric v3 (contract v1.3), dev cohort: {len(scored)} of {len(refs)} references scored "
              f"({len(refs) - len(scored)} disputed). Judge {JUDGE_CONFIG_HASH}, references {REF_CONFIG_HASH}. "
              f"Baseline `{args.baseline}`.", "",
              "| Arm | Correct (95% CI) | Δ correct vs baseline (95% CI) | Severe (95% CI) | Δ severe | Unsupported | Missing req. | Speaker ok | Mode ok | Mean words |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, rows in arms.items():
        lo, hi = ci([rate(rows, ok, d) for d in draws])
        slo, shi = ci([rate(rows, sev, d) for d in draws])
        dc = ds = ""
        if base:
            a, b = ci([rate(rows, ok, d) - rate(base, ok, d) for d in draws])
            dc = f"{rate(rows, ok) - rate(base, ok):+.1%} ({a:+.1%}–{b:+.1%})"
            ds = f"{rate(rows, sev) - rate(base, sev):+.1%}"
        words = sum(len(r["answer"].split()) for r in rows) / len(rows)
        lines.append(f"| {arm} | {rate(rows, ok):.1%} ({lo:.1%}–{hi:.1%}) | {dc} | {rate(rows, sev):.1%} ({slo:.1%}–{shi:.1%}) | {ds} | "
                     f"{rate(rows, lambda r: r['judge']['unsupported']):.1%} | {rate(rows, lambda r: r['judge']['missing_required']):.1%} | "
                     f"{rate(rows, lambda r: r['judge']['speaker_ok']):.1%} | {rate(rows, lambda r: r['judge']['mode_ok']):.1%} | {words:.1f} |")
    lines += ["", "Severe types (answers with each type):", "", "| Arm | " + " | ".join(SEVERE_TYPES) + " |",
              "|---|" + "---|" * len(SEVERE_TYPES)]
    for arm, rows in arms.items():
        c = Counter(t for r in rows for t in set(r["judge"]["severe_types"]) if r["judge"]["severe"])
        lines.append(f"| {arm} | " + " | ".join(str(c[t]) for t in SEVERE_TYPES) + " |")
    items = {i["id"]: i for i in jsonl(BENCH / "mode-suite.jsonl")}
    mode_rows = []
    for path in sorted(BENCH.glob("judged-mode-*.jsonl")):
        rows = jsonl(path)
        if len(rows) != len(items) or any(
                r["key"] != hashlib.sha256(json.dumps([items[r["id"]]["messages"], items[r["id"]]["must"], r["answer"],
                                                       MODE_JUDGE_HASH]).encode()).hexdigest() for r in rows):
            continue  # incomplete or judged under another mode-judge configuration
        name = path.stem.replace("judged-mode-", "")
        system, prompt = name.rsplit("-", 1)
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[items[r["id"]]["kind"]].append(r["judge"])
        cells = " | ".join(f"{sum(j['mode_correct'] for j in by_kind[k])}/{len(by_kind[k])}" for k in MODE_KINDS)
        n = len(rows)
        mode_rows.append(f"| {system}:{prompt} | {sum(r['judge']['mode_correct'] for r in rows)}/{n} | {cells} | "
                         f"{sum(r['judge']['factual_error'] for r in rows)}/{n} | {sum(r['judge']['style_ok'] for r in rows)}/{n} |")
    if mode_rows:
        lines += ["", f"Mode suite ({len(items)} items; judge {MODE_JUDGE_HASH}): mode correct, by kind, factual errors, style ok", "",
                  "| Arm | Mode correct | " + " | ".join(MODE_KINDS) + " | Factual errors | Style ok |",
                  "|---|---|" + "---|" * len(MODE_KINDS) + "---|---|", *mode_rows]
    text = "\n".join(lines)
    (BENCH / "report-dev.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["references", "refreeze", "mode-suite", "challenge", "rechallenge", "run", "judge", "core-build", "core", "report"])
    parser.add_argument("--cohort", choices=["dev", "reserve", "mode"], default="dev")
    parser.add_argument("--arm", help="system:prompt, prompt in short13|full13|short|full")
    parser.add_argument("--kind", choices=["nemotron", "openai"], default="nemotron")
    parser.add_argument("--url")
    parser.add_argument("--checkpoint")
    parser.add_argument("--env-file", default=str(ROOT.parent / "zorana-research-internal/.env"))
    parser.add_argument("--openai-model", default="gpt-5.6-terra")
    parser.add_argument("--baseline", default="v2i328:short")
    parser.add_argument("--seed", type=int, default=99, help="core-build seed")
    parser.add_argument("--show", type=int, default=8, help="core: failures to print")
    parser.add_argument("--temperature", type=float, default=0.0, help="run: sampling temperature for diagnostic arms")
    parser.add_argument("--arms", help="judge only these arms (comma-separated system:prompt)")
    parser.add_argument("--recheck-all", action="store_true", help="refreeze: re-verify every row, not only rule 10-11 hits")
    parser.add_argument("--promotion", action="store_true", help="allow the once-only reserve scoring")
    parser.add_argument("--size", type=int, default=40, help="challenge: number of train transcripts")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.command == "references" and args.cohort == "mode":
        raise SystemExit("references are for dev or reserve")
    command = {"references": references, "refreeze": refreeze, "mode-suite": mode_suite, "challenge": challenge,
               "rechallenge": rechallenge, "run": run_arm, "judge": judge, "core": core}.get(args.command)
    if command:
        asyncio.run(command(args))
    elif args.command == "core-build":
        core_build(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
