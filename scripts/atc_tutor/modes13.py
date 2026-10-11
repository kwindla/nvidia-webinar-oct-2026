"""v1.3 mode-boundary and chat targets (contract docs/atc-tutor-contract-v13.md).

Families (each family's conversations stay in one split):
  mode_pair    60 pairs: an ATC transmission phrased as a question or request (explain it)
               and a user question to the tutor about the same concept (answer it)
  term_number  80 questions about ATC terms or number conversions, answered in one or two
               short plain sentences
  chat         v1.3 versions of the v2 chat categories (session start, small talk,
               capabilities, thanks, off-topic, repeat that, follow-ups, mixed history,
               tutor-directed instructions); ATC explanations in their history follow v1.3
Opus writes in batches; GPT-5.6 Terra critiques every conversation for facts, mode and
style (pass / fix / drop), so the writer and the checker are different model families.
Writes artifacts/atc-tutor/v13/targets-mode.jsonl and targets-chat.jsonl.

Usage: modes13.py [--scale 1.0]
"""

import argparse
import asyncio
import json

import data13
from bakeoff13 import terra_client, terra_tool
from opus import call_tool, client, ledger_summary

TURN = {"type": "object", "required": ["role", "content", "train", "style"], "properties": {
    "role": {"enum": ["user", "assistant"]}, "content": {"type": "string"}, "train": {"type": "boolean"},
    "style": {"enum": ["atc", "chat"], "description": "atc for an ATC explanation, chat otherwise"}}}
BATCH_TOOL = {"name": "conversations", "input_schema": {
    "type": "object", "required": ["conversations"], "properties": {"conversations": {"type": "array", "items": {
        "type": "object", "required": ["category", "turns"],
        "properties": {"category": {"type": "string"}, "turns": {"type": "array", "items": TURN}}}}}}}
CRITIC_TOOL = {"name": "critique", "input_schema": {
    "type": "object", "required": ["problems", "verdict", "fixed_turns"],
    "properties": {"problems": {"type": "array", "items": {"type": "string"}},
                   "verdict": {"enum": ["pass", "fix", "drop"]},
                   "fixed_turns": {"type": "array", "items": TURN, "description": "All turns, corrected, if verdict is fix; else empty."}}}}

KINDS = {
    "mode_pair": (6, 10, """Write {n} PAIRS (2·{n} conversations). In each pair, conversation one: the user
says a realistic ATC transmission that is phrased as a question or a request ("confirm you are
able", "are you aware", "do you have the traffic in sight", "say intentions", "request …"); the
assistant EXPLAINS it in one v1.3 sentence (style atc). Conversation two: the user asks the TUTOR a
direct question about the same concept ("what does say intentions mean?"); the assistant ANSWERS in
one or two short plain sentences (style chat). Set category to "mode_pair:<pair number>" on both.
Vary callsigns, airports, values and phrasing; use only made-up transmissions."""),
    "term_number": (8, 10, """Write {n} conversations where the user asks about an ATC term, abbreviation or
number convention (flight levels to feet, QNH, squawk codes, runway numbering, heading and
compass directions, frequencies, FATO, RVR, CTR, TMA, holding point, wilco, roger, affirm, say
again, expect, cleared, line up and wait, go around, readback, touch and go, VFR, IFR, ILS, RNP).
The assistant answers in one or two short plain sentences, exact and correct, numbers in words
(style chat). Category "term_number"."""),
    "chat": (12, 10, """Write {n} conversations across these categories (category field = the name):
session_start (assistant-first greeting, exactly the contract's session-start line), small_talk,
capabilities, thanks_bye, off_topic (brief answer or polite decline, then invite a phrase),
repeat_that (an ATC transmission, its v1.3 explanation, then the user asks to repeat; restate more
simply), follow_up (an ATC transmission, its v1.3 explanation, then a follow-up question; answer
briefly with no invented operational detail), mixed_history (two earlier transmissions with v1.3
explanations, then a third; explain only the latest), instruction_like (the user tells the tutor
"stand by", "roger, next one", "say again" meaning the tutor; respond conversationally). Only the
final assistant turn has train true; earlier turns train false. ATC explanations anywhere follow
the v1.3 contract exactly."""),
}

WRITER_SYSTEM = f"""You write voice-conversation training examples for a small ATC phrase tutor
model. Follow this contract exactly:

{data13.CONTRACT}

All transmissions you write are invented; never copy real recordings. Assistant chat turns
are short, warm, plain spoken English with numbers in words and no markdown."""
CRITIC_SYSTEM = f"""You check voice-conversation training examples for a small ATC phrase tutor
model against this contract:

{data13.CONTRACT}

Check every trained assistant turn: factual correctness (ATC terms, conversions, phraseology),
mode (an ATC transmission is explained, never answered; a question to the tutor is answered),
v1.3 style (callsigns as heard, plain verbs, numbers in words, short, no markdown), and that
untrained history turns are also correct. Verdict pass if all is right, fix with corrected
turns, or drop."""


async def main(args):
    opus, terra, gate = client(), terra_client(), asyncio.Semaphore(args.concurrency)
    counts = {"mode": 0, "chat": 0}

    async def batch(kind, k, n, brief):
        async with gate:
            try:
                result = await call_tool(opus, stage="modes13_write", system=WRITER_SYSTEM, tool=BATCH_TOOL,
                                         user=brief.format(n=n) + f"\nBatch {k}: make it different from other batches.")
            except Exception as error:
                print("write error", kind, k, error)
                return
        for j, conv in enumerate(result["conversations"]):
            family = conv["category"] if kind == "mode_pair" else f"{kind}-{k}-{j}"
            family = f"{kind}-{k}-{family.split(':')[-1]}" if kind == "mode_pair" else family
            async with gate:
                try:
                    review = await terra_tool(terra, CRITIC_SYSTEM, CRITIC_TOOL, json.dumps(conv["turns"]))
                except Exception as error:
                    print("critic error", kind, k, j, error)
                    continue
            if review["verdict"] == "drop" or (review["verdict"] == "fix" and not review["fixed_turns"]):
                continue
            turns = review["fixed_turns"] if review["verdict"] == "fix" else conv["turns"]
            if not any(t["train"] and t["role"] == "assistant" for t in turns):
                continue
            source = "mode" if kind in ("mode_pair", "term_number") else "chat"
            data13.append(data13.V13 / f"targets-{source}.jsonl", {
                "id": f"{kind}-{k}-{j}", "family": family, "category": conv["category"].split(":")[0],
                "turns": turns, "critic": {"verdict": review["verdict"], "problems": review["problems"]}})
            counts[source] += 1

    jobs = []
    for kind, (batches, n, brief) in KINDS.items():
        for k in range(max(1, round(batches * args.scale))):
            jobs.append(batch(kind, k, n, brief))
    await asyncio.gather(*jobs)
    print(counts, "ledger:", {k: round(v["usd"], 2) for k, v in ledger_summary().items() if k.startswith("modes13")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=12)
    asyncio.run(main(parser.parse_args()))
