"""On-policy DPO pairs for the v1.3 round-2 model (plan: "v1.3 DPO declaration").

sample  four answers per training-side input (split13 `train` real transcripts and the core
        inputs) from the policy endpoint, v1.3 short prompt, temperature 0.7
label   GPT-5.6 Terra critiques every distinct answer against the contract, the input's
        approved v1.3 target (as a reference), telephony entries and core facts; the
        deterministic checks also run. Labeling uses Terra so the Opus evaluator is not
        the reward.
stage   one pair per input: rejected = a sample with a severe error or a failed check;
        chosen = a passing sample (on-policy) if any, else the approved target. Component
        split for validation. Writes a chosen-only PhoneData export for prep_phone_data
        (dpo/<label>/{train,validation}.chat.jsonl) and pairs.jsonl.
pairs   dpo_build.pairs: join prep's rendered prompt with chosen/rejected completions.

Usage:
  dpo13.py sample --url URL --checkpoint PATH
  dpo13.py label
  dpo13.py stage --label atc-dpo-v13
  (upload, prep_phone_data, download rendered files, as in README "Optional DPO stage")
  dpo13.py pairs --label atc-dpo-v13
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter

from dotenv import dotenv_values
from openai import AsyncOpenAI

import data13
import split13
from bakeoff13 import terra_client, terra_tool
from check import normalize
from opus import OUT, ROOT

sys.path.insert(0, str(ROOT))
from tutor_prompts import SHORT13_PROMPT  # noqa: E402

V13 = data13.V13
SAMPLES = V13 / "dpo-samples.jsonl"
LABELS = V13 / "dpo-labels.jsonl"
DPO = OUT / "dpo"
K = 4


def inputs():
    """Training-side inputs with their approved (restyled) target."""
    train_ids = {i for i, r in split13.load().items() if r["part13"] == "train"}
    rows = []
    for source in ("real", "core"):
        for r in data13.jsonl(V13 / f"targets-{source}-r.jsonl"):
            if source == "real" and r["id"] not in train_ids:
                continue
            rows.append({"id": r["id"], "source": source, "component": r["component"] if source == "real" else r["id"],
                         "transmission": r["transmission"], "target": r["explanation"], "facts": r.get("facts")})
    return rows


async def sample(args):
    api = AsyncOpenAI(base_url=args.url.rstrip("/") + "/v1", timeout=900,
                      api_key=dotenv_values(ROOT.parent / "zorana-research-internal/.env")["GB_SERVING_API_KEY"])
    done = {(r["id"], r["k"]) for r in data13.jsonl(SAMPLES)}
    gate = asyncio.Semaphore(args.concurrency)

    async def one(row, k):
        async with gate:
            for attempt in range(3):
                try:
                    response = await api.chat.completions.create(
                        model="gb-phone", temperature=0.7, max_tokens=200,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        messages=[{"role": "system", "content": SHORT13_PROMPT},
                                  {"role": "user", "content": row["transmission"].lower()}])
                    choice = response.choices[0]
                    if choice.finish_reason != "stop" or not (choice.message.content or "").strip():
                        return
                    data13.append(SAMPLES, {"id": row["id"], "k": k, "answer": choice.message.content.strip(),
                                            "checkpoint": args.checkpoint})
                    return
                except Exception:  # noqa: BLE001 - infrastructure retry
                    await asyncio.sleep(10 * (attempt + 1))

    await asyncio.gather(*(one(r, k) for r in inputs() for k in range(K) if (r["id"], k) not in done))
    print("samples:", len(data13.jsonl(SAMPLES)))


async def label(args):
    by_id = {r["id"]: r for r in inputs()}
    done = {(r["id"], normalize(r["answer"])) for r in data13.jsonl(LABELS)}
    todo = {}
    for s in data13.jsonl(SAMPLES):
        key = (s["id"], normalize(s["answer"]))
        if key not in done and key not in todo:
            todo[key] = s
    terra, gate = terra_client(), asyncio.Semaphore(args.concurrency)

    async def one(s):
        row = by_id[s["id"]]
        problems = data13.checks(row["transmission"], s["answer"])
        extra = {"checked_facts": row["facts"]} if row["facts"] else {}
        async with gate:
            try:
                review = await terra_tool(terra, data13.CRITIC_SYSTEM, data13.CRITIC_TOOL, json.dumps({
                    "transmission": row["transmission"], "telephony_table_entries": data13.table_hits(row["transmission"]),
                    **extra, "approved_reference_explanation": row["target"], "candidate": s["answer"],
                    "deterministic_problems": problems}))
            except Exception as error:  # noqa: BLE001
                print("label error", s["id"], error)
                return
        data13.append(LABELS, {"id": s["id"], "answer": s["answer"], "checks": problems, "verdict": review["verdict"],
                               "severe": review["severe"], "problems": review["problems"]})

    await asyncio.gather(*(one(s) for s in todo.values()))
    labels = data13.jsonl(LABELS)
    print("labels:", len(labels), Counter(l["verdict"] for l in labels), "severe:", sum(l["severe"] for l in labels))


STYLE_ONLY = ("digits:", "quotes or markdown", "more than one sentence", "uses 'callsign ends in'", "too long")


def semantic(problems):
    """Deterministic problems that change meaning (not pure form)."""
    return [p for p in problems if not p.startswith(STYLE_ONLY)]


def near_duplicate(a, b):
    """Same meaning-bearing tokens and at most three other words different."""
    from restyle13 import KEEP
    wa, wb = set(re.findall(r"[a-z'-]+", a.lower())), set(re.findall(r"[a-z'-]+", b.lower()))
    meaning = lambda t: sorted(w.lower() for w in KEEP.findall(t))
    return meaning(a) == meaning(b) and len(wa ^ wb) <= 3 and \
        ("pilot" in wa) == ("pilot" in wb) and ("controller" in wa) == ("controller" in wb) and ("speaker" in wa) == ("speaker" in wb)


def clean(label):
    return label["verdict"] == "pass" and not label["severe"] and not label["problems"] and not label["checks"]


def family(row):
    if row["source"] == "real":
        return row["component"]
    kind, n = row["id"].split("-")[1], int(row["id"].split("-")[2])
    import core13
    return f"core:{kind}:{n % len(core13.TEMPLATES[kind][1])}"


async def relabel_targets(rows):
    """Fallback chosen targets get the same frozen Terra criterion as sampled answers."""
    path = V13 / "dpo-target-labels.jsonl"
    done = {r["id"] for r in data13.jsonl(path)}
    terra, gate = terra_client(), asyncio.Semaphore(24)

    async def one(row):
        problems = data13.checks(row["transmission"], row["target"])
        extra = {"checked_facts": row["facts"]} if row["facts"] else {}
        async with gate:
            try:
                review = await terra_tool(terra, data13.CRITIC_SYSTEM, data13.CRITIC_TOOL, json.dumps({
                    "transmission": row["transmission"], "telephony_table_entries": data13.table_hits(row["transmission"]),
                    **extra, "candidate": row["target"], "deterministic_problems": problems}))
            except Exception as error:  # noqa: BLE001
                print("target label error", row["id"], error)
                return
        data13.append(path, {"id": row["id"], "answer": row["target"], "checks": problems, "verdict": review["verdict"],
                             "severe": review["severe"], "problems": review["problems"]})

    await asyncio.gather(*(one(r) for r in rows if r["id"] not in done))
    return {r["id"]: r for r in data13.jsonl(path)}


def stage(args):
    import hashlib
    from export13 import held_out_index
    from split import masked
    rows = inputs()
    exact, templates = held_out_index()
    core_check = OUT / "bench/v13/core-check.jsonl"
    exact |= {normalize(r["transmission"]) for r in data13.jsonl(core_check)}
    stats = Counter()
    # One input per unique transmission; held-out text never enters.
    unique = {}
    for row in rows:
        key = normalize(row["transmission"])
        template = masked(row["transmission"])
        if key in exact or (len(template.split()) >= 6 and template in templates):
            stats["heldout_text"] += 1
            continue
        if key in unique:
            stats["duplicate_text"] += 1
            continue
        unique[key] = row
    labels = {}
    for l in data13.jsonl(LABELS):
        labels.setdefault(l["id"], []).append(l)
    rng = random.Random(20261010)
    candidates, need_target = [], []
    for row in sorted(unique.values(), key=lambda r: r["id"]):
        ls = labels.get(row["id"], [])
        if not ls:
            stats["no_samples"] += 1
            continue
        severe = [l for l in ls if l["severe"] or l["verdict"] == "drop" or semantic(l["checks"])]
        style = [l for l in ls if l not in severe and l["checks"]]
        if not severe and not style:
            stats["no_rejected"] += 1
            continue
        good = [l for l in ls if clean(l)]
        candidates.append((row, severe, style, good))
        if not good:
            need_target.append(row)
    target_labels = asyncio.run(relabel_targets(need_target))
    kept, style_pairs = [], 0
    for row, severe, style, good in candidates:
        if good:
            chosen, origin = rng.choice(sorted(good, key=lambda l: l["answer"]))["answer"], "on_policy"
        elif row["id"] in target_labels and clean(target_labels[row["id"]]):
            chosen, origin = row["target"], "approved_target"
        else:
            stats["chosen_unusable"] += 1
            continue
        pool = severe or style
        rejected = rng.choice(sorted(pool, key=lambda l: l["answer"]))
        if not severe:
            if style_pairs >= 0.15 * len(candidates):
                stats["style_only_capped"] += 1
                continue
            style_pairs += 1
        if normalize(chosen) == normalize(rejected["answer"]):
            stats["identical"] += 1
            continue
        if near_duplicate(chosen, rejected["answer"]):  # inconsistent labels on equivalent wording: noise
            stats["near_duplicate"] += 1
            continue
        failure = "severe" if rejected["severe"] else "drop" if rejected["verdict"] == "drop" else \
            "semantic_check" if semantic(rejected["checks"]) else "style_only"
        stats[f"chosen:{origin}"] += 1
        stats[f"failure:{failure}"] += 1
        stats[f"source:{row['source']}"] += 1
        kept.append({"id": row["id"], "source": row["source"], "component": family(row), "transmission": row["transmission"],
                     "chosen": chosen, "rejected": rejected["answer"], "failure": failure, "chosen_origin": origin,
                     "rejected_problems": rejected["problems"] + rejected["checks"]})
    components = sorted({p["component"] for p in kept})
    rng.shuffle(components)
    val, count = set(), 0
    for component in components:
        if count >= max(16, len(kept) // 10):
            break
        val.add(component)
        count += sum(1 for p in kept if p["component"] == component)
    out = DPO / args.label
    out.mkdir(parents=True, exist_ok=False)
    for name, part in (("validation.chat.jsonl", [p for p in kept if p["component"] in val]),
                       ("train.chat.jsonl", [p for p in kept if p["component"] not in val])):
        (out / name).write_text("".join(json.dumps({
            "conversation_id": f"{args.label}-{p['id']}", "scenario": f"atc-dpo13/{p['failure']}", "tools": [],
            "messages": [{"role": "system", "content": SHORT13_PROMPT, "train": False},
                         {"role": "user", "content": p["transmission"].lower(), "train": False},
                         {"role": "assistant", "content": p["chosen"], "train": True}]}) + "\n" for p in part))
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in kept))
    lengths = [len(p["chosen"].split()) for p in kept], [len(p["rejected"].split()) for p in kept]
    stats.update(kept=len(kept), validation=sum(p["component"] in val for p in kept),
                 train=sum(p["component"] not in val for p in kept))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"stats": dict(stats), "parent_checkpoint": "/data/checkpoints/atc-tutor-v13ar-s2-sft/hf-iter486",
                "prompt": "short13", "temperature": 0.7, "samples_per_input": K,
                "mean_words": {"chosen": sum(lengths[0]) / len(kept), "rejected": sum(lengths[1]) / len(kept)},
                "hashes": {"samples": sha(SAMPLES), "labels": sha(LABELS), "target_labels": sha(V13 / "dpo-target-labels.jsonl"),
                           "split13": sha(split13.SPLIT), "telephony": sha(OUT / "telephony-v13.json"),
                           "pairs": sha(out / "pairs.jsonl")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sample", "label", "stage", "pairs"])
    parser.add_argument("--url")
    parser.add_argument("--checkpoint")
    parser.add_argument("--label")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.command == "pairs":
        import dpo_build
        dpo_build.DPO = DPO
        dpo_build.pairs(args)
    elif args.command == "stage":
        stage(args)
    else:
        asyncio.run(globals()[args.command](args))


if __name__ == "__main__":
    main()
