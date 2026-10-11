"""Round-3 data: speech-recognition noise, fragments, emergency squawks, acknowledgements, meta-talk.

The live rehearsal (2026-10-10 19:00) showed the round-2 model inventing values on
misrecognized or cut-off input ("qankis seven five zero zero" -> three invented
frequencies; "roger / one" -> "frequency one megahertz"), calling squawk 7600 the general
emergency code, missing "wilco" in "we welco", and explaining the user's own meta-talk
("I'm saying qantas qantas") as a radio call.

Commands (all write under artifacts/atc-tutor/v13/):
  corrupt   Opus turns training-side transmissions (core inputs and split13 train rows)
            into realistic recognizer output: mishearings, dropped or merged words, or a
            fragment. -> noisy-inputs.jsonl
  special   template inputs with facts: emergency squawks and acknowledgement words in
            their recognizer spellings. -> special-inputs.jsonl
  targets   Opus writes, GPT-5.6 Terra critiques (data13.write_and_audit). Noisy rows get the
            clean original only as a hint, with the rule that nothing the transcription
            lost may be stated; a deterministic check drops any target that states a
            digit run present in the original but absent from the transcription.
            -> targets-noisy.jsonl, targets-special.jsonl
  meta      meta-talk chat conversations (the user corrects or repeats what they said,
            asks a question the recognizer garbled, thanks the tutor), Terra-critiqued.
            -> targets-meta.jsonl
  contrast  round 4: chat-versus-radio contrast conversations (greetings, small talk and
            "is this on" said to the tutor, the same courtesy words inside a callsign
            transmission, and short term/conversion answers), Terra-critiqued.
            -> targets-contrast.jsonl
  check     the frozen noisy check: 80 corrupted core-check items (held out from
            training) -> bench/v13/noisy-check.jsonl
  score     run a served arm on the noisy check at temperature 0 and score it: the
            deterministic lost-value check plus an Opus judge (invents / faithful).
            -> bench/v13/noisy-<arm>.jsonl

Usage: noisy13.py corrupt | special | targets --source noisy|special | meta | contrast | check
       noisy13.py score --arm NAME:short13 --url https://…modal.run
"""

import argparse
import asyncio
import json
import random
import re

import sys

import data13
import restyle13
import split13
import telephony
from bakeoff13 import terra_client, terra_tool
from core13 import SQUAWK_MEANING, render
from modes13 import BATCH_TOOL, CRITIC_SYSTEM as MODE_CRITIC, CRITIC_TOOL as MODE_CRITIC_TOOL
from modes13 import WRITER_SYSTEM as MODE_WRITER
from opus import OUT, call_tool, client, ledger_summary
from values import Sampler, spoken_digits

V13 = data13.V13
sys.path.insert(0, str(OUT.parent.parent))
BENCH = OUT / "bench/v13"
KINDS = ("mishear", "mishear", "mishear", "drop_merge", "fragment")
CORRUPT_TOOL = {"name": "recognized", "input_schema": {
    "type": "object", "required": ["recognized", "changes"],
    "properties": {"recognized": {"type": "string", "description": "What the recognizer output."},
                   "changes": {"type": "string", "description": "Which words changed and how."}}}}
CORRUPT_SYSTEM = """You simulate a speech recognizer listening to a person (often not a pilot)
reading an air traffic control transmission aloud into a laptop microphone. Given the clean
transmission and a corruption kind, output what the recognizer would plausibly produce:
lowercase words, numbers as spoken words, no punctuation.

Kinds:
- mishear: one to three words become sound-alike words or non-words, as real recognizers
  do with aviation vocabulary: "squawking" -> "qankis", "cathay" -> "yasy", "wilco" ->
  "we welco" or "will co", "affirm" -> "a firm", "roger" -> "rodger" or "roger that",
  "cleared" -> "clear" or "cle", "until lima lima" -> "intelma", "niner" -> "nine her",
  "flight level" -> "fight level", "descend" -> "the send", "heading" -> "heading" (often
  fine), callsign words garbled, a digit word misheard as another ("five" -> "fine",
  "two" -> "to", "four" -> "for", "eight" -> "a").
- drop_merge: one or two words are dropped or two words merge or split ("line up" ->
  "lineup", "contact tower" -> "contact our"), possibly including a value word.
- fragment: only part was captured, as when the speaker paused and the turn ended: just
  the start (often the callsign and first word or two) or just the end ("one two zero
  expect ils runway two five right"), or an acknowledgement with one stray word ("roger
  one").

Make it realistic, not random. Keep everything else exactly as given."""

NOISE_FACT = ("The transmission is speech-recognizer output. The speaker most likely said: \"{clean}\". "
              "Use that only to recognize words the transcription still supports (a misheard word you "
              "can identify from its sound, with 'probably' where it is a guess). Never state a value, "
              "callsign part or instruction that the transcription itself lost: if a value or clause is "
              "missing or cut off, say only what was heard, or that the rest is unclear or cut off.")

SPECIAL_TEMPLATES = {
    "emergency": (120, ["{cs} squawk {sqe}", "squawk {sqe} {cs}", "squawking {sqe} {cs}", "{cs} squawking {sqe}",
                        "{cs} confirm squawking {sqe}", "{cs} reset squawk {sqe}", "mayday mayday mayday {cs} squawking {sqe}",
                        "{cs} squawk {sqe} contact {f}", "{cs} we are squawking {sqe}", "{cs} select {sqe}"]),
    "ack": (90, ["wilco {cs}", "{cs} wilco", "will co {cs}", "we welco {cs}", "roger {cs}", "rodger", "{cs} roger",
                 "roger that {cs}", "affirm {cs}", "a firm {cs}", "affirmative", "negative {cs}", "unable {cs}",
                 "say again {cs}", "say a gain", "standby {cs}", "copy that {cs}", "{cs} copied",
                 "roger wilco {cs}", "{cs} affirm"]),
}
ACK_FACTS = [(r"\bwil ?co\b|\bwe welco\b|\bwill co\b", "wilco = will comply (understood and will do it)"),
             (r"\brod?ger\b", "roger = message received and understood (not a promise to comply)"),
             (r"\ba firm\b|\baffirm(ative)?\b", "affirm = yes"), (r"\bnegative\b", "negative = no"),
             (r"\bunable\b", "unable = cannot comply"), (r"\bsay a ?gain\b", "say again = please repeat"),
             (r"\bstand ?by\b", "standby = wait"), (r"\bcop(y|ied)\b", "copy = heard and understood")]

META_BRIEF = """Write {n} conversations where the user's last turn is MEANT FOR THE TUTOR, not a radio call,
even though it may contain ATC words or recognizer errors. Category field = the name:
- said_correction: after a tutor explanation, the user says what they actually said ("no, I said
  cathay two three", "that was cathay", "I'm saying qantas qantas", "I said wilco"); the tutor
  acknowledges in a short chat sentence and, if the corrected transmission is clear from the
  history, restates its meaning with the corrected words (one v1.3 sentence), else invites a repeat.
- garbled_question: the user asks the tutor a question about ATC language with recognizer
  errors ("why do pilosa niner insted of nine", "what do dose line up and wait mean", "no I'm
  asking a question about the lingo"); the tutor recognizes the question and answers it in one
  or two short plain sentences.
- thanks: the user thanks or says goodbye to the tutor ("thank you", "thanks good day", "great
  thanks") right after an explanation; a short warm chat reply, not an ATC explanation. (Contrast:
  "thank you good day" as an ATC transmission with a callsign is explained.)
- retry: the user signals a retry or an aborted start ("wait", "sorry let me try again", "um so",
  "hold on"); a short reply inviting them to go ahead.
Earlier turns are a realistic session (a transmission with its v1.3 explanation). Only the final
assistant turn has train true; style chat (atc only if it restates a transmission)."""

CONTRAST_BRIEF = """Write {n} conversations, five PAIRS, that teach when words are meant for the tutor and when they
are part of a radio call. Category field = "<kind>:<pair letter>" with kinds:
- to_tutor: the user's last turn is talk to the tutor, with no callsign or station: greetings
  ("good morning", "hi there, how are you", "hey, how's it going"), small talk ("did you have a
  nice weekend", "busy day?"), checks ("is this on", "can you hear me", "are you there", "is the
  radio on"), thanks or praise. Sometimes it is the first user turn, sometimes it follows an ATC
  explanation. The tutor replies briefly and warmly in chat style and invites a phrase; it never
  says "the speaker".
- in_radio: the SAME courtesy words inside a realistic ATC transmission with a callsign or
  station ("good morning vienna radar lufthansa four two one climbing flight level two four zero",
  "speedbird two one good day contact london one two seven decimal one"); the tutor explains it
  in one v1.3 sentence (style atc), keeping the greeting brief.
Pair each to_tutor conversation with an in_radio one that reuses its courtesy words. In two of the
five pairs, instead make the to_tutor turn a short question to the tutor about a term or number
("what does wilco mean", "what is QNH one zero one three in inches", "how many feet is flight
level one two zero"), answered correctly in one or two short plain sentences without repetition.
Only the final assistant turn has train true."""


def jsonl(path):
    return data13.jsonl(path)


def digit_runs(text):
    words = text.replace("niner", "nine").split()
    runs, run = [], []
    for w in words + ["."]:
        if w in data13.DIGIT:
            run.append(w)
        else:
            if run:
                runs.append(" ".join(run))
            run = []
    return runs


SOUND_ALIKE = {"for": "four", "to": "two", "too": "two", "fine": "five", "won": "one", "tree": "three",
              "ate": "eight", "fife": "five", "nein": "nine", "sex": "six", "zeroe": "zero"}  # recoverable digit mishearings


def lost_values(clean, noisy, answer):
    """Digit runs of two or more digits in the original that the transcription lost but the answer states."""
    from values import words_int
    heard = " ".join(SOUND_ALIKE.get(w, w) for w in noisy.replace("niner", "nine").replace("nine her", "nine").split())
    noisy_text, low = f" {heard} ", f" {answer.lower().replace('-', ' ')} "
    problems = []
    for run in digit_runs(clean):
        if len(run.split()) < 2 or f" {run} " in noisy_text:
            continue
        number = int("".join(str(data13.DIGIT[w]) for w in run.split()))
        forms = {run, words_int(number).replace("-", " ")}
        if any(f" {f} " in low for f in forms):
            problems.append(f"states '{run}', which the transcription lost")
    return problems


async def corrupt_rows(rows, stage, seed):
    api, gate, rng = client(), asyncio.Semaphore(16), random.Random(seed)
    for r in rows:
        r["noise_kind"] = rng.choice(KINDS)

    async def run(r):
        async with gate:
            try:
                out = await call_tool(api, stage=stage, system=CORRUPT_SYSTEM, tool=CORRUPT_TOOL, effort="medium",
                                      user=json.dumps({"clean": r["transmission"], "kind": r["noise_kind"]}))
            except Exception as error:
                print("error", r["id"], error)
                return None
        noisy = " ".join(re.sub(r"[^a-z' ]", " ", out["recognized"].lower()).split())
        if not noisy or noisy == r["transmission"]:
            return None
        return {**r, "clean": r["transmission"], "transmission": noisy, "changes": out["changes"]}

    return [x for x in await asyncio.gather(*(run(r) for r in rows)) if x]


async def corrupt(args):
    rng = random.Random(20261011)
    core = jsonl(V13 / "core-inputs.jsonl")
    rng.shuffle(core)
    real_targets = {r["id"]: r for r in jsonl(V13 / "targets-real-r.jsonl")}
    quarantined = split13.quarantine()
    real = [{"id": r["id"], "component": r["component"], "transmission": r["transmission"], "source": "real"}
            for r in real_targets.values() if r.get("ambiguity") == "clear" and r["component"] not in quarantined
            and len(r["transmission"].split()) >= 4]
    rng.shuffle(real)
    picks = [{**r, "id": f"nz-{r['id']}", "parent_id": r["id"]} for r in core[:args.core] + real[:args.real]]
    rows = await corrupt_rows(picks, "noisy13_corrupt", 20261011)
    (V13 / "noisy-inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "noisy inputs;", {k: sum(r["noise_kind"] == k for r in rows) for k in set(KINDS)})
    for r in rows[:: max(1, len(rows) // 12)]:
        print(f"  [{r['noise_kind']}] {r['clean']}  ->  {r['transmission']}")


def special(args):
    rng, sampler = random.Random(20261012), Sampler(20261012)
    rows = []
    for kind, (count, templates) in SPECIAL_TEMPLATES.items():
        for n in range(count):
            dialect = rng.choice(["icao_europe", "icao_europe", "icao_australia", "faa"])
            text, facts = render(rng, sampler, templates[n % len(templates)], dialect)
            if kind == "emergency":
                code = next(c for c in SQUAWK_MEANING if spoken_digits(c) in text)
                facts = [f for f in facts if not f.startswith("squawk")] + [
                    f"squawk {spoken_digits(code)} is {SQUAWK_MEANING[code]}"]
            else:
                facts = [meaning for pattern, meaning in ACK_FACTS if re.search(pattern, text)]
            rows.append({"id": f"sp-{kind}-{n:03d}", "component": f"special:{kind}", "kind": kind,
                         "transmission": text, "facts": facts, "source": "special"})
    (V13 / "special-inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "special inputs")
    for r in rows[::23]:
        print(" ", r["transmission"], "|", "; ".join(r["facts"]))


async def targets(args):
    items = jsonl(V13 / f"{args.source}-inputs.jsonl")
    out, drops = V13 / f"targets-{args.source}.jsonl", V13 / f"targets-{args.source}-drops.jsonl"
    done = {r["id"] for r in jsonl(out)} | {r["id"] for r in jsonl(drops)}
    apis, gate, table = {"opus": client(), "terra": terra_client()}, asyncio.Semaphore(args.concurrency), telephony.load()

    async def run(item):
        facts = list(item.get("facts") or [])
        if args.source == "noisy":
            facts.append(NOISE_FACT.format(clean=item["clean"]))
        async with gate:
            try:
                result = await data13.write_and_audit(apis, item["transmission"], f"noisy13_{args.source}", "opus", facts)
            except Exception as error:
                print("error", item["id"], error)
                return
        if not result.get("drop"):
            text, _ = restyle13.airline_style(item["transmission"], result["explanation"], table)
            lost = lost_values(item["clean"], item["transmission"], text) if args.source == "noisy" else []
            if lost or data13.checks(item["transmission"], text):
                result = {"drop": True, "history": result["history"], "late_problems": lost or data13.checks(item["transmission"], text)}
            else:
                result["explanation"] = text
        data13.append(drops if result.get("drop") else out, {**item, **result})

    await asyncio.gather(*(run(i) for i in items if i["id"] not in done))
    print(f"{args.source}: kept {len(jsonl(out))}, dropped {len(jsonl(drops))}")
    print("ledger:", {k: round(v["usd"], 2) for k, v in ledger_summary().items() if k.startswith("noisy13")})


async def meta(args, brief=META_BRIEF, name="meta"):
    opus, terra, gate = client(), terra_client(), asyncio.Semaphore(12)
    path = V13 / f"targets-{name}.jsonl"

    async def batch(k):
        async with gate:
            result = await call_tool(opus, stage=f"noisy13_{name}_write", system=MODE_WRITER, tool=BATCH_TOOL,
                                     user=brief.format(n=10) + f"\nBatch {k}: make it different from other batches.")
        for j, conv in enumerate(result["conversations"]):
            async with gate:
                review = await terra_tool(terra, MODE_CRITIC, MODE_CRITIC_TOOL, json.dumps(conv["turns"]))
            if review["verdict"] == "drop" or (review["verdict"] == "fix" and not review["fixed_turns"]):
                continue
            turns = review["fixed_turns"] if review["verdict"] == "fix" else conv["turns"]
            if any(t["train"] and t["role"] == "assistant" for t in turns):
                family = f"{name}-{k}-{conv['category'].split(':')[-1]}" if name == "contrast" else f"{name}-{k}-{j}"
                data13.append(path, {"id": f"{name}-{k}-{j}", "family": family, "category": conv["category"],
                                     "turns": turns, "critic": {"verdict": review["verdict"], "problems": review["problems"]}})

    await asyncio.gather(*(batch(k) for k in range(args.batches)))
    print(f"{name}:", len(jsonl(path)), "ledger:", {k: round(v["usd"], 2) for k, v in ledger_summary().items() if k.startswith("noisy13")})


async def check(args):
    path = BENCH / "noisy-check.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; the noisy check is frozen")
    rng = random.Random(20261013)
    items = jsonl(BENCH / "core-check.jsonl")
    rng.shuffle(items)
    picks = [{"id": f"nc-{r['id']}", "transmission": r["transmission"], "kind": r["kind"]} for r in items[:args.n]]
    rows = await corrupt_rows(picks, "noisy13_check_corrupt", 20261013)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(path, len(rows), {k: sum(r["noise_kind"] == k for r in rows) for k in set(KINDS)})


JUDGE_TOOL = {"name": "verdict", "input_schema": {
    "type": "object", "required": ["invents", "faithful", "note"],
    "properties": {
        "invents": {"type": "boolean", "description": "The answer states a value, callsign part, instruction or "
                    "meaning that the transcription does not support (a guess not marked as probable counts)."},
        "faithful": {"type": "boolean", "description": "The answer conveys what the transcription does support, "
                     "recovering clearly identifiable mishearings, and says what is unclear or cut off."},
        "note": {"type": "string"}}}}
JUDGE_SYSTEM = """You grade an ATC phrase tutor's one-sentence explanation of a speech-recognizer transcript.
You see the transcript, what the speaker most likely said (the tutor did NOT see this), and the
answer. Judge only against what a careful listener could recover from the transcript itself:
sound-alike words may be recovered (ideally marked "probably"); values, callsign parts or
clauses missing from the transcript must not be supplied. Squawk 7500 is unlawful interference,
7600 radio failure, 7700 emergency; wilco means will comply; roger means received."""
JUDGE_HASH = __import__("hashlib").sha256((JUDGE_SYSTEM + json.dumps(JUDGE_TOOL)).encode()).hexdigest()[:16]


async def score(args):
    from dotenv import dotenv_values
    from openai import AsyncOpenAI
    from tutor_prompts import SHORT13_PROMPT
    items = jsonl(BENCH / "noisy-check.jsonl")
    system, prompt_name = args.arm.split(":")
    assert prompt_name == "short13"
    path = BENCH / f"noisy-{system}-{prompt_name}.jsonl"
    api = AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", timeout=900,
                      api_key=dotenv_values(str(OUT.parent.parent.parent / "zorana-research-internal/.env"))["GB_SERVING_API_KEY"])
    opus, gate = client(), asyncio.Semaphore(16)

    async def run(item):
        async with gate:
            response = await api.chat.completions.create(
                model="gb-phone", temperature=0, max_tokens=200,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "system", "content": SHORT13_PROMPT}, {"role": "user", "content": item["transmission"]}])
            answer = (response.choices[0].message.content or "").strip()
            verdict = await call_tool(opus, stage="noisy13_judge", system=JUDGE_SYSTEM, tool=JUDGE_TOOL, effort="medium",
                                      user=json.dumps({"transcript": item["transmission"], "most_likely_said": item["clean"],
                                                       "answer": answer}))
        lost = lost_values(item["clean"], item["transmission"], answer) + data13.implausible(answer)
        return {"id": item["id"], "arm": args.arm, "judge_hash": JUDGE_HASH, "answer": answer,
                "lost_values": lost, **verdict}

    rows = await asyncio.gather(*(run(i) for i in items))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(f"{args.arm}: noisy check n={n}; invents {sum(r['invents'] for r in rows)}, "
          f"lost-value statements {sum(bool(r['lost_values']) for r in rows)}, faithful {sum(r['faithful'] for r in rows)}"
          f" (judge {JUDGE_HASH})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["corrupt", "special", "targets", "meta", "contrast", "check", "score"])
    parser.add_argument("--source", choices=["noisy", "special"])
    parser.add_argument("--core", type=int, default=380)
    parser.add_argument("--real", type=int, default=170)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--arm")
    parser.add_argument("--url")
    args = parser.parse_args()
    if args.command == "special":
        return special(args)
    asyncio.run({"corrupt": corrupt, "targets": targets, "meta": meta, "check": check, "score": score,
                 "contrast": lambda a: meta(a, CONTRAST_BRIEF, "contrast")}[args.command](args))


if __name__ == "__main__":
    main()
