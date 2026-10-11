"""v1.3 split: reserve whole devpool recording components for a once-only promotion check.

Writes artifacts/atc-tutor/v13/split.jsonl (one row per transcript with `part13`):
  test       the consumed final-test split (never trained on, never re-used to select)
  selection  the old selection split (development cohort)
  devdiag    the old devdiag partition (development cohort)
  reserve    newly withheld devpool components (promotion cohort, scored once)
  train      remaining devpool rows (v1.3 real-input training side)

Components holding stress-case demonstrations are forced to `train`. Selection is
stratified by airport (whole components, shuffled with a fixed seed) and never looks
at any model output. `quarantine()` is the set of components whose descendants
must not appear in v1.3 training.
"""

import json
import random
from collections import defaultdict

from opus import OUT

V13 = OUT / "v13"
SPLIT = V13 / "split.jsonl"
SEED = 20261010
RESERVE_TARGET = 250
FORCED_TRAIN = {  # stress-case sources (docs/atc-tutor-contract-v13.md)
    "LSGS_SION_Ground_Control_121_7MHz_20210502_114120",
    "YSSY_SYDNEY_Tower_120_5MHz_20210604_221813",
    "LSZB_BERN_Tower_121_0MHz_20210423_091530",
    "LSGS_SION_Ground_Control_121_7MHz_20210503_141438",
    "LKPR_RUZYNE_Tower_134_560MHz_20201027_213741",
    "LSGS_SION_Ground_Control_121_7MHz_20210503_152011",
}


def jsonl(path):
    return [json.loads(line) for line in path.open()]


def build():
    rows = jsonl(OUT / "split.jsonl")
    partition = {r["id"]: r["partition"] for r in jsonl(OUT / "dev-partition.jsonl")}
    pool = defaultdict(list)
    for r in rows:
        if partition.get(r["id"]) == "devpool" and r["component"] not in FORCED_TRAIN:
            pool[r["component"]].append(r)
    by_airport = defaultdict(list)
    for comp, members in sorted(pool.items()):
        by_airport[members[0]["airport"]].append(comp)
    total = sum(len(m) for m in pool.values())
    rng = random.Random(SEED)
    reserve = set()
    for airport, comps in sorted(by_airport.items()):
        quota = RESERVE_TARGET * sum(len(pool[c]) for c in comps) / total
        rng.shuffle(comps)
        taken = 0
        for comp in comps:
            if taken >= quota:
                break
            reserve.add(comp)
            taken += len(pool[comp])
    V13.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with SPLIT.open("w") as f:
        for r in rows:
            if r["split"] == "test":
                part = "test"
            elif r["split"] == "selection":
                part = "selection"
            elif partition.get(r["id"]) == "devdiag":
                part = "devdiag"
            elif r["component"] in reserve:
                part = "reserve"
            else:
                part = "train"
            counts[part] += 1
            f.write(json.dumps({**r, "part13": part}) + "\n")
    print(dict(counts), "reserve components:", len(reserve))


def load():
    return {r["id"]: r for r in jsonl(SPLIT)}


def quarantine():
    """Components whose rows or descendants must never reach v1.3 training."""
    return {r["component"] for r in load().values() if r["part13"] != "train"}


if __name__ == "__main__":
    if SPLIT.exists():
        raise SystemExit(f"{SPLIT} exists; the v1.3 split is immutable")
    build()
