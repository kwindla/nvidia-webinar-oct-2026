"""General-chat and multi-turn slice for the ATC tutor (Claude Opus 5.5).

Categories cover the session-start greeting (assistant-first, always with a system
row), small talk, capability questions, follow-up questions about ATC terms,
"repeat that", thanks/goodbye, off-topic requests, and multi-turn histories where
only the final assistant turn is trained.

Usage:
  uv run python scripts/atc_tutor/chat.py --per-category 30   # chat.jsonl
  uv run python scripts/atc_tutor/chat.py --audit             # chat.jsonl -> chat-audited.jsonl
"""

import argparse
import asyncio
import json

from opus import OUT, ROOT, call_tool, client, ledger_summary

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
CATEGORIES = {
    "session_start": "The conversation has just started and the user has not spoken. The assistant greets with exactly the session-start line from the contract. Turns: one assistant turn only.",
    "small_talk": "The user makes small talk (how are you, nice day, who are you). One user turn, one short warm assistant reply that invites an aviation phrase.",
    "capabilities": "The user asks what the assistant can do or how to use it. One user turn, one short reply.",
    "term_question": "The user asks what an ATC word or concept means (QNH, squelch, wilco vs roger, flight level, squawk, holding point, ILS, go around). One user turn, one or two short plain sentences.",
    "repeat_that": "Multi-turn: user says an ATC transmission, assistant explains it in one sentence, then the user asks to repeat or rephrase ('say that again', 'what?', 'slower please'); the final assistant turn restates the explanation more simply. Only the final assistant turn is trained.",
    "follow_up": "Multi-turn: user says an ATC transmission, assistant explains, then the user asks a follow-up ('why would they do that?', 'what does readback mean?'); the final assistant turn answers briefly without inventing operational details. Only the final assistant turn is trained.",
    "thanks_bye": "The user thanks the assistant or says goodbye. One short reply.",
    "off_topic": "The user asks something unrelated (weather today, a recipe, a math question). The assistant answers very briefly or politely declines if it cannot know, then invites an aviation phrase.",
    "mixed_history": "Multi-turn with two earlier ATC transmissions and explanations as history, then a third transmission; the final assistant turn explains only the latest transmission. Only the final assistant turn is trained.",
    "instruction_like": "The user speaks directly to the assistant with an instruction that resembles ATC wording but is meant for the tutor ('say again', 'stand by a second', 'roger that, next one'). Decide sensibly: a direct request to the tutor is handled conversationally; an obvious ATC phrase is explained.",
}
TOOL = {"name": "record_conversations", "input_schema": {
    "type": "object", "required": ["conversations"],
    "properties": {"conversations": {"type": "array", "items": {
        "type": "object", "required": ["turns"],
        "properties": {"turns": {"type": "array", "items": {
            "type": "object", "required": ["role", "content", "train"],
            "properties": {
                "role": {"enum": ["user", "assistant"]},
                "content": {"type": "string"},
                "train": {"type": "boolean"},
            }}}}}}}}}
SYSTEM = f"""You write short voice-conversation training examples for an ATC phrase tutor
that speaks through text-to-speech. Follow this contract; ATC explanations must obey it
exactly, and every assistant reply is plain spoken text, brief, with numbers in words.

{CONTRACT}

User turns are speech-recognition text: lowercase, no punctuation, sometimes with small
recognition slips. Vary users, phrasing, topics, callsigns and values widely. Mark train
true only on the assistant turn(s) the category says to train."""


AUDIT_TOOL = {"name": "audit", "input_schema": {
    "type": "object", "required": ["acceptable", "problems"],
    "properties": {"acceptable": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}}
AUDIT_SYSTEM = f"""You audit voice-conversation training examples for an ATC phrase tutor
against this contract:

{CONTRACT}

Judge every assistant turn marked train=true in its conversation context. It is
acceptable only if any ATC explanation is faithful to the visible transmission
(speech act, speaker only when evident, values, conditions, expectations vs
clearances, negation, nothing invented, no guessed repairs, abbreviated callsigns not
expanded), the reply is brief plain spoken text with numbers in words, and it fits the
conversation."""


async def audit():
    rows = [json.loads(l) for l in (OUT / "chat.jsonl").open()]
    api = client()
    gate = asyncio.Semaphore(16)
    kept, problems = [], 0

    async def run(row):
        nonlocal problems
        async with gate:
            result = await call_tool(api, stage="chat_audit", system=AUDIT_SYSTEM, tool=AUDIT_TOOL,
                                     user=json.dumps({"turns": row["turns"]}))
        if result["acceptable"]:
            kept.append({**row, "audit": result})
        else:
            problems += 1

    await asyncio.gather(*(run(r) for r in rows))
    (OUT / "chat-audited.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"chat conversations kept {len(kept)} of {len(rows)}; dropped {problems}")
    print("ledger:", json.dumps(ledger_summary()))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        await audit()
        return
    api = client()
    rows = []

    async def run(category, description, index):
        result = await call_tool(api, stage="chat", system=SYSTEM, tool=TOOL, user=json.dumps({
            "category": category, "instructions": description, "count": args.batch, "batch": index}))
        for conversation in result.get("conversations", []):
            turns = [[t["role"], t["content"].strip(), bool(t["train"]) and t["role"] == "assistant"]
                     for t in conversation["turns"]]
            if not turns or not any(t[2] for t in turns):
                continue
            if category != "session_start" and turns[0][0] != "user":
                continue
            # The session-start row needs a system prompt (assistant-first rows render only with one).
            rows.append({"slice": "chat", "category": category, "turns": turns,
                         "format": "short" if category == "session_start" else None})

    await asyncio.gather(*(run(c, d, i) for c, d in CATEGORIES.items()
                           for i in range(max(1, args.per_category // args.batch))))
    (OUT / "chat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} chat conversations")
    print("ledger:", json.dumps(ledger_summary()))


if __name__ == "__main__":
    asyncio.run(main())
