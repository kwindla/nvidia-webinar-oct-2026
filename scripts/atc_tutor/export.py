"""Export audited ATC tutor rows as PhoneData chat JSONL for prep_phone_data.

Rows: {conversation_id, scenario, tools: [], messages: [{role, content, train}]}.
Each target gets one prompt format (short / full / none; none only for user-first
rows). Validation: every descendant of ~5% of seeds (familiar shapes) plus every
row of ~3% of rare shapes (unseen compositions); noise and contrast rows follow
their seed. Gates: every target is re-checked with check.py; value-guess hedges
and unadjudicated needs_review rows are dropped; any conversation whose user text
matches a selection/test transcript (exact, or slot-masked with six or more
tokens, the frozen split rule) is dropped; labels are immutable.

Usage: uv run python scripts/atc_tutor/export.py --label atc-tutor-v2
Writes artifacts/atc-tutor/export/<label>/{train,validation}.chat.jsonl,
provenance.jsonl and manifest.json.
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from opus import OUT, ROOT

sys.path.insert(0, str(ROOT))
from tutor_prompts import FULL_PROMPT, TRAINED_PROMPT_SHA256_PREFIX  # noqa: E402
from tutor_prompts import TUTOR_PROMPT as SHORT_PROMPT  # noqa: E402

CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()
# The frozen contract and the shared prompt module must agree.
assert re.search(r"## Short tutor prompt.*?```\n(.*?)```", CONTRACT, re.S).group(1).strip() == SHORT_PROMPT
GREETING = "Let's practice aviation radio calls. Say an aviation phrase, and I'll briefly explain what it means."


def prompt_hashes():
    hashes = {name: hashlib.sha256(text.encode()).hexdigest()[:16]
              for name, text in (("short", SHORT_PROMPT), ("full", FULL_PROMPT))}
    if hashes != TRAINED_PROMPT_SHA256_PREFIX:
        raise SystemExit(f"prompts differ from the trained prompts: {hashes}")
    return hashes


def conversation(cid, scenario, fmt, turns):
    """turns: list of (role, content, train). Assistant-first rows always keep a system row."""
    messages = []
    if fmt != "none" or turns[0][0] == "assistant":
        messages.append({"role": "system", "content": FULL_PROMPT if fmt == "full" else SHORT_PROMPT, "train": False})
    messages += [{"role": role, "content": content, "train": train} for role, content, train in turns]
    return {"conversation_id": cid, "scenario": scenario, "tools": [], "messages": messages}


VALUE_GUESS = re.compile(
    r"\b(probably|likely|possibly|perhaps|presumably)\b[^.]{0,80}\b(zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|twenty|thirty|hundred|thousand)\b", re.I)


def load(name):
    path = OUT / name
    return [json.loads(l) for l in path.open()] if path.exists() else []


def main():
    from check import check_row
    from restyle import restyle
    from split import masked, normalize

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--restyle", action="store_true", help="contract v1.2 registration style (v3+)")
    parser.add_argument("--repairs", help="repair slice file in artifacts/atc-tutor (repairs-<arm>.jsonl)")
    parser.add_argument("--targeted", action="store_true", help="include audited-targeted.jsonl (v4 heading slice)")
    parser.add_argument("--exclude-devdiag", action="store_true",
                        help="exclude the frozen devdiag partition's lineage and text (v3+)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = OUT / "export" / args.label
    if out.exists():
        raise SystemExit(f"{out} exists; export labels are immutable, choose a new label")

    split_rows = [json.loads(l) for l in (OUT / "split.jsonl").open()]
    real_all = frozenset(normalize(r["text"]) for r in split_rows)
    partition = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "dev-partition.jsonl").open()} \
        if (args.exclude_devdiag or args.repairs) else {}
    devdiag_ids = {i for i, r in partition.items() if r["partition"] == "devdiag"} if args.exclude_devdiag else set()
    devpool_ids = {i for i, r in partition.items() if r["partition"] == "devpool"}
    held_rows = [r for r in split_rows if r["split"] in ("selection", "test") or r["id"] in devdiag_ids]
    held_exact = {normalize(r["text"]) for r in held_rows}
    held_masked = {masked(r["text"]) for r in held_rows if len(masked(r["text"]).split()) >= 6}
    # Repairs are real devpool transcripts; only selection/test/devdiag texts are off limits for them.
    real_heldout = frozenset(held_exact)
    drops = Counter()

    def leaks(text):
        """Frozen split rule: exact normalized text, or a slot-masked template of six or more tokens."""
        template = masked(text)
        return normalize(text) in held_exact or (len(template.split()) >= 6 and template in held_masked)

    def regate(row, slice_name):
        if row.get("audit", {}).get("verdict", "pass") != "pass":
            drops[f"{slice_name}:audit_drop"] += 1
            return False
        if row.get("seed_id") in devdiag_ids:
            drops[f"{slice_name}:devdiag_lineage"] += 1
            return False
        result = check_row(row, real_heldout if slice_name == "repair" else real_all)
        if result["status"] == "reject":
            drops[f"{slice_name}:check_reject"] += 1
            return False
        if result["status"] == "needs_review" and not (row.get("audit") or {}).get("adjudication"):
            drops[f"{slice_name}:unadjudicated_review"] += 1
            return False
        if VALUE_GUESS.search(row["explanation"]):
            drops[f"{slice_name}:value_guess"] += 1
            return False
        return True

    targets = [{**t, "slice": "shape"} for t in load("audited.jsonl") if regate(t, "shape")]
    targets += [{**t, "slice": "contrast"} for t in load("audited-contrast.jsonl") if regate(t, "contrast")]
    targeted = [t for t in load("audited-targeted.jsonl") if regate(t, "targeted")] if args.targeted else []
    if args.targeted and not targeted:
        raise SystemExit("--targeted requested but no targeted rows were accepted")
    repairs = []
    if args.repairs:
        texts = {i: normalize(r["text"]) for i, r in partition.items()}
        for r in load(args.repairs):
            if r["seed_id"] not in devpool_ids or normalize(r["utterance"]) != texts.get(r["seed_id"]):
                drops["repair:not_devpool"] += 1
                continue
            if regate({**r, "values_used": [], "bundle": {}}, "repair"):
                repairs.append(r)
        if not repairs:
            raise SystemExit("--repairs requested but no repair rows were accepted")
    noise = []
    for row in load("noise-audited.jsonl"):
        candidate = {**row, "values_used": [], "bundle": {}}
        if regate(candidate, "noise"):
            noise.append(row)
    chats = load("chat-audited.jsonl")
    if not chats:
        raise SystemExit("chat-audited.jsonl missing; run chat.py --audit")

    # Splits: unseen-composition holdout from rare shapes; familiar holdout by seed.
    # Every descendant of a held-out seed (shape, contrast, noise) goes to validation.
    shape_sizes = Counter(t["shape"] for t in targets)
    rare = sorted(shape for shape, size in shape_sizes.items() if size <= 3)
    seeds = sorted({t["seed_id"] for t in targets})
    val_shapes = set(rng.sample(rare, max(1, round(0.03 * len(shape_sizes)))))
    val_seeds = set(rng.sample(seeds, max(1, round(0.05 * len(seeds)))))
    shape_of_seed = {t["seed_id"]: t["shape"] for t in targets}

    def holdout(seed_id):
        if shape_of_seed.get(seed_id) in val_shapes:
            return "unseen_shape"
        return "familiar_shape" if seed_id in val_seeds else None

    train, validation, provenance = [], [], []
    formats = ["short", "full", "none"]

    def add(cid, scenario, fmt, turns, held, meta):
        users = [content for role, content, _ in turns if role == "user"]
        if any(leaks(text) for text in users):
            drops[f"{meta['slice']}:heldout_text_match"] += 1
            return
        if args.restyle:
            restyled, last_user = [], ""
            for role, content, train_flag in turns:
                if role == "user":
                    last_user = content
                elif role == "assistant":
                    new = restyle(last_user, content)
                    if new is None:
                        drops[f"{meta['slice']}:restyle_unrewritable"] += 1
                        return
                    if new != content and meta["slice"] == "chat":
                        drops["chat:restyle_would_change"] += 1
                        return
                    if new != content and train_flag:
                        recheck = check_row({"utterance": last_user, "explanation": new,
                                             "values_used": [], "bundle": {}}, frozenset())
                        if recheck["status"] == "reject":
                            drops[f"{meta['slice']}:restyle_check_reject"] += 1
                            return
                        drops[f"{meta['slice']}:restyled"] += 1
                    content = new
                restyled.append((role, content, train_flag))
            turns = restyled
        (validation if held else train).append(conversation(cid, scenario, fmt, turns))
        provenance.append({"conversation_id": cid, "split": held or "train", "format": fmt, **meta})

    for index, row in enumerate(targets):
        fmt = formats[index % 3]
        # The live fine-tuned ASR emits uppercase finals; train some robustness to case.
        user_text = row["utterance"].upper() if rng.random() < 0.15 else row["utterance"]
        add(f"{args.label}-{row['slice']}-{index:05d}",
            "atc-tutor/" + row["shape"].split("#")[0].replace("|", "/")[:80], fmt,
            [("user", user_text, False), ("assistant", row["explanation"], True)],
            holdout(row["seed_id"]),
            {"seed_id": row["seed_id"], "shape": row["shape"], "dialect": row["dialect"], "slice": row["slice"]})
    for index, row in enumerate(noise):
        add(f"{args.label}-noise-{index:05d}", "atc-tutor/noise", formats[index % 3],
            [("user", row["utterance"], False), ("assistant", row["explanation"], True)],
            holdout(row["seed_id"]), {"seed_id": row["seed_id"], "slice": "noise"})
    for index, row in enumerate(targeted):
        add(f"{args.label}-targeted-{index:05d}", "atc-tutor/targeted/heading", formats[index % 3],
            [("user", row["utterance"], False), ("assistant", row["explanation"], True)],
            "targeted_validation" if rng.random() < 0.05 else None,
            {"seed_id": row["seed_id"], "slice": "targeted"})
    for index, row in enumerate(repairs):
        # Repairs are real dev transcripts (never selection/test); each row is its own seed.
        add(f"{args.label}-repair-{index:05d}", f"atc-tutor/repair/{row['failure']}", formats[index % 3],
            [("user", row["utterance"], False), ("assistant", row["explanation"], True)],
            holdout(row["seed_id"]), {"seed_id": row["seed_id"], "slice": "repair", "failure": row["failure"]})
    seen = set()
    for index, row in enumerate(chats):
        key = json.dumps(row["turns"]).lower()
        if key in seen:
            drops["chat:duplicate"] += 1
            continue
        seen.add(key)
        held = "chat_validation" if rng.random() < 0.05 else None
        add(f"{args.label}-chat-{index:05d}", f"atc-tutor/chat/{row['category']}",
            row.get("format") or formats[index % 3], [tuple(t) for t in row["turns"]], held,
            {"slice": "chat", "category": row["category"]})

    # Cross-split guard: no validation user text (exact or masked) may appear in training.
    def user_keys(rows):
        keys = set()
        for r in rows:
            for m in r["messages"]:
                if m["role"] != "user":
                    continue
                keys.add((r["conversation_id"], normalize(m["content"])))
                template = masked(m["content"])
                if len(template.split()) >= 6:
                    keys.add((r["conversation_id"], "M:" + template))
        return keys
    train_keys = {k for _, k in user_keys(train)}
    leaked = sorted({cid for cid, k in user_keys(validation) if k in train_keys})
    if leaked:
        validation = [r for r in validation if r["conversation_id"] not in leaked]
        provenance = [p for p in provenance if p["conversation_id"] not in leaked]
        drops["validation:overlaps_train"] += len(leaked)
    if not train or not validation:
        raise SystemExit("empty split")

    if args.repairs and not any(p["slice"] == "repair" and p["split"] == "train" for p in provenance):
        raise SystemExit("--repairs requested but no repair rows reached the training split")
    sources = [(n, OUT / n) for n in ("audited.jsonl", "audited-contrast.jsonl", "noise-audited.jsonl",
                                      "chat-audited.jsonl", "dev-partition.jsonl") + ((args.repairs,) if args.repairs else ())
                + (("audited-targeted.jsonl",) if args.targeted else ())]
    sources += [("target-contract.md", ROOT / "docs/atc-tutor-target-contract.md"),
                ("restyle.py", ROOT / "scripts/atc_tutor/restyle.py"), ("check.py", ROOT / "scripts/atc_tutor/check.py")]
    source_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest()[:16] for name, path in sources if path.is_file()}
    # Write into a temporary directory and rename, so a failed export leaves no partial label.
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    final_out, out = out, staging
    for name, rows in (("train.chat.jsonl", train), ("validation.chat.jsonl", validation)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "provenance.jsonl").write_text("".join(json.dumps(p) + "\n" for p in provenance))
    count = lambda rows: sum(m["train"] for r in rows for m in r["messages"])
    manifest = {
        "label": args.label,
        "train_conversations": len(train), "validation_conversations": len(validation),
        "train_targets": count(train), "validation_targets": count(validation),
        "formats": dict(Counter(p["format"] for p in provenance)),
        "slices": dict(Counter(p["slice"] for p in provenance)),
        "validation_kinds": dict(Counter(p["split"] for p in provenance)),
        "drops": dict(sorted(drops.items())),
        "prompt_hashes": prompt_hashes(),
        "split_manifest_sha256": hashlib.sha256((OUT / "split.jsonl").read_bytes()).hexdigest(),
        "flags": {"restyle": args.restyle, "repairs": args.repairs, "targeted": args.targeted,
                  "exclude_devdiag": args.exclude_devdiag},
        "source_sha256": source_sha256,
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                  for name in ("train.chat.jsonl", "validation.chat.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(final_out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
