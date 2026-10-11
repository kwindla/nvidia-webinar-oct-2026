"""Writer bake-off for v1.3 targets: Claude Opus 5.5 vs GPT-5.6 Terra on the same inputs.

60 training-side real transcripts (split13 `train`, fixed seed, 40 of them long or
garbled-looking). Both writers get the same v1.3 contract, writer instructions and
telephony entries (data13.WRITER_SYSTEM). Every answer is then critiqued by both
families (data13.CRITIC_SYSTEM) and run through data13.checks. A blind pairwise sample
of 30 goes to the owner (artifacts/atc-tutor/v13/bakeoff-pairs.jsonl).

Usage: bakeoff13.py run | report | pairs
"""

import argparse
import asyncio
import json
import random
from collections import Counter

from dotenv import dotenv_values
from openai import AsyncOpenAI

import data13
import split13
from opus import ROOT, call_tool, client, strict

OUT = data13.V13 / "bakeoff.jsonl"
PAIRS = data13.V13 / "bakeoff-pairs.jsonl"
TERRA = "gpt-5.6-terra"


def sample():
    rows = sorted((r for r in split13.load().values() if r["part13"] == "train"), key=lambda r: r["id"])
    rng = random.Random(20261010)
    hard = [r for r in rows if len(r["text"].split()) >= 14 or any(w in r["text"] for w in ("spk", " uh ", "ne french"))]
    easy = [r for r in rows if r not in hard]
    return rng.sample(hard, 40) + rng.sample(easy, 20)


def terra_client():
    return AsyncOpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"], timeout=600)


async def terra_tool(api, system, tool, user):
    response = await api.chat.completions.create(
        model=TERRA, reasoning_effort="high", max_completion_tokens=16000,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": tool["name"], "strict": True, "schema": strict(tool["input_schema"])}})
    usage = response.usage
    with (data13.V13.parent / "terra-ledger.jsonl").open("a") as ledger:
        ledger.write(json.dumps({"model": response.model, "input": usage.prompt_tokens,
                                 "output": usage.completion_tokens,
                                 "reasoning": getattr(usage.completion_tokens_details, "reasoning_tokens", None)}) + "\n")
    return json.loads(response.choices[0].message.content)


async def run(args):
    data13.V13.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in data13.jsonl(OUT)}
    opus, terra, gate = client(), terra_client(), asyncio.Semaphore(args.concurrency)

    async def one(row):
        text = row["text"]
        hits = data13.table_hits(text)
        user = json.dumps({"transmission": text, "telephony_table_entries": hits})
        async with gate:
            try:
                written = {
                    "opus": (await call_tool(opus, stage="bakeoff13_write", system=data13.WRITER_SYSTEM,
                                             tool=data13.WRITE_TOOL, user=user))["explanation"].strip(),
                    "terra": (await terra_tool(terra, data13.WRITER_SYSTEM, data13.WRITE_TOOL, user))["explanation"].strip(),
                }
                critiques = {}
                for writer, answer in written.items():
                    critic_user = json.dumps({"transmission": text, "telephony_table_entries": hits, "candidate": answer,
                                              "deterministic_problems": data13.checks(text, answer)})
                    critiques[writer] = {
                        "opus": await call_tool(opus, stage="bakeoff13_critic", system=data13.CRITIC_SYSTEM,
                                                tool=data13.CRITIC_TOOL, user=critic_user),
                        "terra": await terra_tool(terra, data13.CRITIC_SYSTEM, data13.CRITIC_TOOL, critic_user),
                        "checks": data13.checks(text, answer)}
            except Exception as error:
                print("error", row["id"], error)
                return
        data13.append(OUT, {"id": row["id"], "transmission": text, "telephony": hits,
                            "answers": written, "critiques": critiques})

    await asyncio.gather(*(one(r) for r in sample() if r["id"] not in done))
    report(args)


def report(args):
    rows = data13.jsonl(OUT)
    print(f"{len(rows)} transcripts")
    for writer in ("opus", "terra"):
        words = [len(r["answers"][writer].split()) for r in rows]
        line = [f"{writer}: mean words {sum(words) / len(words):.1f}",
                f"check problems {sum(bool(r['critiques'][writer]['checks']) for r in rows)}"]
        for critic in ("opus", "terra"):
            verdicts = Counter(r["critiques"][writer][critic]["verdict"] for r in rows)
            severe = sum(r["critiques"][writer][critic]["severe"] for r in rows)
            line.append(f"{critic}-critic pass {verdicts['pass']} fix {verdicts['fix']} drop {verdicts['drop']} severe {severe}")
        print(" | ".join(line))


def pairs(args):
    rows = sorted(data13.jsonl(OUT), key=lambda r: r["id"])
    rng = random.Random(7)
    picked = rng.sample(rows, min(30, len(rows)))
    with PAIRS.open("w") as f:
        for n, r in enumerate(picked, 1):
            order = ["opus", "terra"]
            rng.shuffle(order)
            f.write(json.dumps({"id": f"p{n:03d}", "n": n, "sourceId": r["id"], "transmission": r["transmission"],
                                "a": r["answers"][order[0]], "b": r["answers"][order[1]], "aWriter": order[0],
                                "bWriter": order[1]}) + "\n")
    print(PAIRS, len(picked))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run", "report", "pairs"])
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        globals()[args.cmd](args)


if __name__ == "__main__":
    main()
