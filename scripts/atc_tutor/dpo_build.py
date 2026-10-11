"""Build NeMo-RL DPO pairs from a tuned policy's own devpool errors (optional arm).

Construction follows docs/reviews/atc-tutor-v3-results-astra.md ("one bounded DPO
experiment"):
- rejected = the policy's own answer on a devpool transcript that the frozen-rubric
  judge marked wrong (never an endpoint error or empty answer);
- chosen = a candidate-blind repair of that transcript that passed check.py and an
  adjudicator against the frozen reference (mine.py repair --arm <policy arm>);
- seeds whose lineage is in the SFT corpus's validation split are excluded, devdiag,
  selection and test never enter (devpool only);
- both completions share one prompt rendered by the real prep_phone_data template
  (short tutor prompt, no-think prefix): `stage` writes a chosen-only PhoneData export
  for prep; `pairs` joins prep's rendered `input` with chosen/rejected completions.

Usage:
  uv run python scripts/atc_tutor/dpo_build.py stage --policy-arm v3i324-short --corpus atc-tutor-v3 --label atc-dpo-v1
  (prep_phone_data --dataset-label atc-dpo-v1-stage --out-label atc-dpo-v1-rendered, then
   modal volume get training.jsonl, validation.jsonl and rows_meta.jsonl into
   artifacts/atc-tutor/dpo/atc-dpo-v1/)
  uv run python scripts/atc_tutor/dpo_build.py pairs --label atc-dpo-v1
"""

import argparse
import json
import random
from collections import Counter

from check import normalize
from export import SHORT_PROMPT
from opus import OUT

DPO = OUT / "dpo"


def stage(args):
    """Admit pairs only when joined to an accepted repair and passing the export gates."""
    from check import check_row
    from export import VALUE_GUESS
    from split import masked

    corpus = OUT / "export" / args.corpus
    held_lineage = {p["seed_id"] for p in map(json.loads, (corpus / "provenance.jsonl").open())
                    if p.get("seed_id") and p["split"] != "train"}
    partition = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "dev-partition.jsonl").open()}
    split_rows = [json.loads(l) for l in (OUT / "split.jsonl").open()]
    held = [r for r in split_rows if r["split"] in ("selection", "test")] + \
        [r for r in partition.values() if r["partition"] == "devdiag"]
    held_exact = {normalize(r["text"]) for r in held}
    held_masked = {masked(r["text"]) for r in held if len(masked(r["text"]).split()) >= 6}
    repairs = {r["seed_id"]: r for r in map(json.loads, (OUT / f"repairs-{args.policy_arm}.jsonl").open())}
    outputs = {o["id"]: o for o in map(json.loads, (OUT / "bench" / "v2" / f"outputs-devpool-{args.policy_arm}.jsonl").open())}
    pairs = [json.loads(l) for l in (OUT / f"dpo-pairs-{args.policy_arm}.jsonl").open()]
    stats, seen, kept = Counter(mined=len(pairs)), set(), []
    for p in pairs:
        row, repair, output = partition.get(p["id"]), repairs.get(p["id"]), outputs.get(p["id"])
        reason = None
        if not row or row["partition"] != "devpool" or normalize(row["text"]) != normalize(p["transmission"]):
            reason = "not_devpool_text"
        elif p["id"] in held_lineage:
            reason = "validation_lineage"
        elif p["id"] in seen:
            reason = "duplicate"
        elif not repair or repair["explanation"] != p["chosen"] or repair["audit"]["verdict"] != "pass" \
                or (repair["audit"].get("adjudication") or {}).get("problems"):
            reason = "no_accepted_repair"
        elif not output or output["answer"] != p["rejected"] or output["error"] or output.get("finish_reason") != "stop":
            reason = "rejected_not_policy_output"
        elif output["identity"].get("checkpoint") != args.parent_checkpoint:
            reason = "wrong_parent"
        elif not p["chosen"].strip() or not p["rejected"].strip() or normalize(p["chosen"]) == normalize(p["rejected"]):
            reason = "trivial"
        elif VALUE_GUESS.search(p["chosen"]):
            reason = "value_guess"
        elif check_row({"utterance": p["transmission"], "explanation": p["chosen"], "values_used": [], "bundle": {}},
                       frozenset())["status"] == "reject":
            reason = "chosen_check_reject"
        elif normalize(p["transmission"]) in held_exact or masked(p["transmission"]) in held_masked:
            reason = "heldout_text"
        if reason:
            stats[reason] += 1
            continue
        seen.add(p["id"])
        kept.append({**p, "component": row["component"]})
    # Split by recording component, never by row.
    components = sorted({p["component"] for p in kept})
    rng = random.Random(20261010)
    rng.shuffle(components)
    val_components, count = set(), 0
    for component in components:
        if count >= max(8, len(kept) // 10):
            break
        val_components.add(component)
        count += sum(1 for p in kept if p["component"] == component)
    validation = [p for p in kept if p["component"] in val_components]
    train = [p for p in kept if p["component"] not in val_components]
    if len(train) < 50 or len(validation) < 8:
        raise SystemExit(f"too few pairs after gates: train {len(train)}, validation {len(validation)}")
    out = DPO / args.label
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("validation.chat.jsonl", validation), ("train.chat.jsonl", train)):
        (out / name).write_text("".join(json.dumps({
            "conversation_id": f"{args.label}-{p['id']}", "scenario": f"atc-dpo/{p['failure']}", "tools": [],
            "messages": [{"role": "system", "content": SHORT_PROMPT, "train": False},
                         {"role": "user", "content": p["transmission"], "train": False},
                         {"role": "assistant", "content": p["chosen"], "train": True}]}) + "\n" for p in rows))
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in kept))
    stats.update(kept=len(kept), train=len(train), validation=len(validation))
    stats.update({f"failure:{k}": v for k, v in Counter(p["failure"] for p in kept).items()})
    manifest = {"stats": dict(stats), "parent_checkpoint": args.parent_checkpoint, "policy_arm": args.policy_arm,
                "corpus": args.corpus, "split_components": {"validation": len(val_components)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def pairs(args):
    out = DPO / args.label
    by_id = {f"{args.label}-{p['id']}": p for p in map(json.loads, (out / "pairs.jsonl").open())}
    meta = [json.loads(l) for l in (out / "rows_meta.jsonl").open()]
    for split in ("training", "validation"):
        rendered = [json.loads(l) for l in (out / f"{split}.jsonl").open()]
        ids = [m["conversation_id"] for m in meta if m["split"] == split]
        if len(ids) != len(rendered):
            raise SystemExit(f"{split}: rows_meta and rendered rows disagree ({len(ids)} vs {len(rendered)})")
        rows = []
        for cid, r in zip(ids, rendered):
            pair = by_id[cid]
            if not r["input"].endswith("<think></think>") or not r["output"].endswith("<|im_end|>\n"):
                raise SystemExit("unexpected template boundary in rendered row")
            suffix = r["output"][len(pair["chosen"]):]
            if not r["output"].startswith(pair["chosen"]) or suffix != "<|im_end|>\n":
                raise SystemExit("rendered chosen completion does not match the pair")
            rows.append({"prompt": r["input"], "chosen": r["output"], "rejected": pair["rejected"] + suffix,
                         "id": pair["id"], "failure": pair["failure"]})
        (out / f"{split}.dpo.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(split, len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stage", "pairs"])
    parser.add_argument("--policy-arm")
    parser.add_argument("--corpus")
    parser.add_argument("--parent-checkpoint", default="/data/checkpoints/atc-tutor-v2-sft/hf-iter328")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    stage(args) if args.command == "stage" else pairs(args)


if __name__ == "__main__":
    main()
