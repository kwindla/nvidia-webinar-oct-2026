"""Export v1.3 targets as PhoneData chat JSONL for prep_phone_data.

Sources (artifacts/atc-tutor/v13/): targets-real, targets-synthetic, targets-variant,
targets-mode and targets-chat (multi-turn chat/mode rows; optional). Gates, applied to
every user turn of every conversation:
  quarantine   the source component (real row, synthetic seed, variant parent) must be
               split13 `train`
  held-out     no normalized-exact or six-plus-token slot-masked match with any test,
               selection, devdiag or reserve transcript, or any mode-suite turn
  checks       data13.checks passes on every trained assistant turn
Formats: short13, full13 and none, assigned by a stable hash. Validation is about 5% of
training-side components (all their descendants together) plus a 5% sample of the
chat/mode families. `--arm A` excludes variants; `--arm B` includes them. `--round3` adds
the noisy13.py sources (targets-noisy, targets-special, targets-meta) and holds out the
noisy check. `--round4` also adds targets-contrast (chat-versus-radio pairs).

Usage: export13.py --label atc-tutor-v13a --arm A
Writes artifacts/atc-tutor/export/<label>/{train,validation}.chat.jsonl,
provenance.jsonl and manifest.json. Labels are immutable.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter

import data13
import split13
from opus import OUT, ROOT
from split import masked, normalize

sys.path.insert(0, str(ROOT))
from tutor_prompts import FULL13_PROMPT, SHORT13_PROMPT  # noqa: E402

V13 = data13.V13
SUITE = OUT / "bench/v13/mode-suite.jsonl"
FORMATS = ("short13", "full13", "none")


def stable(key, n):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def held_out_index():
    rows = split13.load().values()
    texts = [r["text"] for r in rows if r["part13"] != "train"]
    if SUITE.exists():
        for item in data13.jsonl(SUITE):
            texts += [t["content"] for t in item.get("turns", []) if t.get("role") == "user"]
            if item.get("transmission"):
                texts.append(item["transmission"])
    exact = {normalize(t) for t in texts}
    # Noisy-check items are corrupted core-check templates, which core training shares by
    # design (as with the core check itself), so they are held out by exact text only.
    exact |= {normalize(r["transmission"]) for r in data13.jsonl(OUT / "bench/v13/noisy-check.jsonl")}
    templates = {masked(t) for t in texts if len(masked(t).split()) >= 6}
    return exact, templates


def conversation(cid, scenario, fmt, turns):
    messages = []
    if fmt != "none" or turns[0][0] == "assistant":
        messages.append({"role": "system", "content": FULL13_PROMPT if fmt == "full13" else SHORT13_PROMPT,
                         "train": False})
    messages += [{"role": role, "content": content, "train": train} for role, content, train in turns]
    return {"conversation_id": cid, "scenario": scenario, "tools": [], "messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--restyled", action="store_true", help="use the owner-style targets (*-r.jsonl) and the core slice")
    parser.add_argument("--round3", action="store_true", help="add the noisy13 sources (implies --restyled)")
    parser.add_argument("--round4", action="store_true", help="round 3 plus the contrast conversations")
    args = parser.parse_args()
    args.round3 = args.round3 or args.round4
    args.restyled = args.restyled or args.round3
    target = OUT / "export" / args.label
    if target.exists():
        raise SystemExit(f"{target} exists; labels are immutable")
    split, quarantined = split13.load(), split13.quarantine()
    exact, templates = held_out_index()
    drops = Counter()

    def leaks(text):
        template = masked(text)
        return normalize(text) in exact or (len(template.split()) >= 6 and template in templates)

    items = []  # (group, provenance, turns)
    sources = ["real", "synthetic"] + (["variant"] if args.arm == "B" else []) + (["core"] if args.restyled else [])
    suffix = "-r" if args.restyled else ""
    seen = set()
    sources += ["noisy", "special"] if args.round3 else []
    for source in sources:
        for r in data13.jsonl(V13 / (f"targets-{source}{suffix}.jsonl" if source not in ("noisy", "special")
                                     else f"targets-{source}.jsonl")):
            if r["id"] in seen:  # resumed runs can overlap; keep the first target per input
                drops[f"{source}:duplicate"] += 1
                continue
            seen.add(r["id"])
            if r["component"] in quarantined:
                drops[f"{source}:quarantine"] += 1
                continue
            if leaks(r["transmission"]):
                drops[f"{source}:heldout_text"] += 1
                continue
            if data13.checks(r["transmission"], r["explanation"]):
                drops[f"{source}:checks"] += 1
                continue
            if source in ("core", "special") or (source == "noisy" and r["component"].startswith("core:")):
                group = r.get("parent_id") or r["id"]  # template rows have no recording lineage
            else:
                group = r["component"]
            items.append((group, {"id": r["id"], "source": source, "component": r["component"],
                                           "parent_id": r.get("parent_id"), "seed_id": r.get("seed_id"),
                                           "ambiguity": r.get("ambiguity")},
                          [("user", r["transmission"], False), ("assistant", r["explanation"], True)]))
    extra = (("meta",) if args.round3 else ()) + (("contrast",) if args.round4 else ())
    for source in ("mode", "chat") + extra:
        for r in data13.jsonl(V13 / (f"targets-{source}{suffix}.jsonl" if source in ("mode", "chat") else f"targets-{source}.jsonl")):
            users = [t["content"] for t in r["turns"] if t["role"] == "user"]
            if any(leaks(u) for u in users):
                drops[f"{source}:heldout_text"] += 1
                continue
            if any(data13.checks("", t["content"]) and t.get("train") and t.get("style") == "atc" for t in r["turns"]):
                drops[f"{source}:checks"] += 1
                continue
            items.append((f"family:{r['family']}", {"id": r["id"], "source": source, "family": r["family"],
                                                    "category": r.get("category")},
                          [(t["role"], t["content"], bool(t.get("train"))) for t in r["turns"]]))

    train, validation, provenance = [], [], []
    for n, (group, meta, turns) in enumerate(sorted(items, key=lambda x: x[1]["id"])):
        is_val = stable(group, 20) == 0
        fmt = FORMATS[stable(meta["id"], 3)]
        if fmt == "none" and turns[0][0] == "assistant":
            fmt = "short13"
        cid = f"{args.label}-{meta['source']}-{n:05d}"
        conv = conversation(cid, f"atc-tutor-v13/{meta['source']}", fmt, turns)
        (validation if is_val else train).append(conv)
        provenance.append({"conversation_id": cid, "split": "validation" if is_val else "train", "format": fmt, **meta})

    # Cross-split guard: no validation user text may appear in training.
    train_users = {normalize(m["content"]) for c in train for m in c["messages"] if m["role"] == "user"}
    leaked = [c for c in validation if any(normalize(m["content"]) in train_users
                                           for m in c["messages"] if m["role"] == "user")]
    validation = [c for c in validation if c not in leaked]
    drops["validation:overlaps_train"] = len(leaked)

    staging = target.with_name(target.name + ".staging")
    staging.mkdir(parents=True)
    files = {}
    for name, rows in (("train.chat.jsonl", train), ("validation.chat.jsonl", validation),
                       ("provenance.jsonl", provenance)):
        body = "".join(json.dumps(r) + "\n" for r in rows)
        (staging / name).write_text(body)
        files[name] = hashlib.sha256(body.encode()).hexdigest()
    manifest = {
        "label": args.label, "arm": args.arm, "restyled": args.restyled, "round3": args.round3, "round4": args.round4, "train_conversations": len(train),
        "validation_conversations": len(validation),
        "sources": Counter(p["source"] for p in provenance if p["split"] == "train"),
        "formats": Counter(p["format"] for p in provenance),
        "drops": drops, "files": files,
        "split13_sha256": hashlib.sha256(split13.SPLIT.read_bytes()).hexdigest(),
        "telephony_sha256": hashlib.sha256((OUT / "telephony-v13.json").read_bytes()).hexdigest(),
        "prompts": {name: hashlib.sha256(text.encode()).hexdigest()[:16]
                    for name, text in (("short13", SHORT13_PROMPT), ("full13", FULL13_PROMPT))},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
