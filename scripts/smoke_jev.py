"""Check Jev on known ATC mismatches, number formatting, and local agreement."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

from asr_comparison import ComparisonSession

CASES = [
    (
        "number error",
        "united four fifty two turn left heading two seven zero",
        "united four fifty two turn left heading two seven zero",
        "united four fifty two turn left heading two one zero",
        "fine-tuned",
    ),
    (
        "omitted clause",
        "descend and maintain five thousand feet",
        "five thousand feet",
        "descend and maintain five thousand feet",
        "base",
    ),
    (
        "equivalent number format",
        "heading two seven zero",
        "heading two seven zero",
        "heading 270",
        "tie",
    ),
    ("identical", "hello", "hello", "hello", "identical"),
]


async def main():
    load_dotenv(".env")
    results = []
    context_cases = [
        (
            "context readback fine-tuned",
            "",
            "descend and maintain five thousand feet",
            "descend and maintain fine thousand feet",
            "fine-tuned",
        ),
        (
            "context readback base",
            "",
            "contact Prague terror",
            "contact Prague tower",
            "base",
        ),
        (
            "no evidence for new number",
            "",
            "heading two seven zero",
            "heading two one zero",
            "insufficient",
        ),
    ]
    histories = {
        "context readback fine-tuned": [
            {
                "role": "assistant",
                "content": "Please read back: descend and maintain five thousand feet.",
            }
        ],
        "context readback base": [
            {
                "role": "assistant",
                "content": "Read back this instruction: contact Prague tower.",
            }
        ],
    }
    for name, reference, first, second, expected in CASES + context_cases:
        c = ComparisonSession(None, context_getter=lambda: histories.get(name, []))
        c.status = {s: "ready" for s in ["fine-tuned", "base"]}
        c.configure({"reference": reference, "context_judging": True})
        turn = c.speech_started()
        c.reserve_commit()
        c.turn_stopped()
        for source, text in [("fine-tuned", first), ("base", second)]:
            c.event(source, {"type": "x.completed", "transcript": text})
            c.event(source, {"type": "input_audio_buffer.committed"})
        if turn.judge["state"] == "pending":
            async with asyncio.timeout(10):
                result = await c._ask_jev(turn)
        else:
            result = turn.judge
        passed = result.get("winner") == expected
        results.append(
            {
                "case": name,
                "expected": expected,
                "passed": passed,
                "result": result,
                "wer": turn.wer,
            }
        )
        print(name, result.get("winner"), "PASS" if passed else "DIFFERS")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/jev-known-winners.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    if not all(r["passed"] for r in results):
        raise AssertionError(
            "Jev differs from expected labels; see artifacts/jev-known-winners.json"
        )


if __name__ == "__main__":
    asyncio.run(main())
