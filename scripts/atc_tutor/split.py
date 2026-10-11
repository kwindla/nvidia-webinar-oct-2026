"""Freeze the dev / selection / final-test split of the ATC preview transcripts.

Groups rows by recording prefix, unions groups that share an exact normalized
transcript or a long slot-masked transcript, forces already-exposed rows into
dev, then assigns whole components to splits stratified by airport.
"""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "datasets/atc02-preview/data/test/metadata.csv"
OUT = ROOT / "artifacts/atc-tutor"
SEED = 20261009
TEST_FRACTION = 0.20
SELECTION_FRACTION = 0.10  # of the non-test rows
MASKED_MIN_WORDS = 6  # shorter masked templates are too generic to imply lineage

# Rows already shown in this project's conversation or reviews: forced to dev.
EXPOSED_IDS = {
    "LZIB_STEFANIK_Tower_118_3MHz_20210429_075711!004",
    "LKPR_RUZYNE_Radar_120_520MHz_20201027_090116!001",
    "LKPR_RUZYNE_Radar_120_520MHz_20201027_090116!002",
}
# Seeds of the random samples printed during dataset profiling.
EXPOSED_SAMPLES = ((1, 8), (7, 25))
# Isolated transmissions named as stress cases in the reviews.
EXPOSED_TEXTS = {"wilco", "xt", "sorry go ahead"}

NUMBER = (
    r"(?:zero|one|two|three|tree|four|five|fife|six|seven|eight|nine|niner|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|decimal|point)"
)
LETTER = (
    r"(?:alfa|alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliett|"
    r"juliet|kilo|lima|mike|november|oscar|papa|quebec|romeo|sierra|tango|"
    r"uniform|victor|whiskey|whisky|x ray|xray|yankee|zulu)"
)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def masked(text):
    text = f" {normalize(text)} "
    text = re.sub(rf"( {LETTER})+( {NUMBER})*", " <ID>", text)
    text = re.sub(rf"( {NUMBER})+", " <N>", text)
    return re.sub(r"\s+", " ", text).strip()


class Components:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    rows = list(csv.DictReader(METADATA.open()))
    exposed = set(EXPOSED_IDS)
    for seed, count in EXPOSED_SAMPLES:
        exposed |= {row["id"] for row in random.Random(seed).sample(rows, count)}
    exposed |= {row["id"] for row in rows if normalize(row["text"]) in EXPOSED_TEXTS}

    prefix = {row["id"]: row["id"].split("!")[0] for row in rows}
    components = Components(set(prefix.values()))
    by_key = defaultdict(set)
    for row in rows:
        by_key["exact:" + normalize(row["text"])].add(prefix[row["id"]])
        template = masked(row["text"])
        if len(template.split()) >= MASKED_MIN_WORDS:
            by_key["masked:" + template].add(prefix[row["id"]])
    for groups in by_key.values():
        first, *rest = sorted(groups)
        for other in rest:
            components.union(first, other)

    members = defaultdict(list)
    for row in rows:
        members[components.find(prefix[row["id"]])].append(row)
    forced_dev = {
        root for root, items in members.items() if any(r["id"] in exposed for r in items)
    }

    rng = random.Random(SEED)
    split_of = {}
    by_airport = defaultdict(list)
    for root, items in members.items():
        if root in forced_dev:
            split_of[root] = "dev"
        else:
            airport = Counter(r["airport"] for r in items).most_common(1)[0][0]
            by_airport[airport].append(root)
    for airport, roots in sorted(by_airport.items()):
        roots.sort()
        rng.shuffle(roots)
        target = TEST_FRACTION * sum(
            len(members[root]) for root in members
            if Counter(r["airport"] for r in members[root]).most_common(1)[0][0] == airport
        )
        taken = 0
        for root in roots:
            if taken < target:
                split_of[root] = "test"
                taken += len(members[root])
            else:
                split_of[root] = "dev"
    dev_roots = sorted(root for root, split in split_of.items() if split == "dev" and root not in forced_dev)
    rng.shuffle(dev_roots)
    dev_rows = sum(len(members[root]) for root, split in split_of.items() if split == "dev")
    taken = 0
    for root in dev_roots:
        if taken >= SELECTION_FRACTION * dev_rows:
            break
        split_of[root] = "selection"
        taken += len(members[root])

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for root, items in sorted(members.items()):
        for row in sorted(items, key=lambda r: r["id"]):
            records.append({
                "id": row["id"],
                "file_name": row["file_name"],
                "text": row["text"],
                "airport": row["airport"],
                "component": root,
                "split": split_of[root],
                "exposed": row["id"] in exposed,
            })
    manifest = OUT / "split.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))

    sizes = Counter(len(items) for items in members.values())
    report = {
        "seed": SEED,
        "rows": len(rows),
        "recording_prefixes": len(set(prefix.values())),
        "components": len(members),
        "largest_component": max(sizes),
        "component_size_histogram": dict(sorted(sizes.items())),
        "exposed_rows": len(exposed),
        "forced_dev_components": len(forced_dev),
        "rows_by_split": dict(Counter(r["split"] for r in records)),
        "rows_by_split_airport": {
            split: dict(Counter(r["airport"] for r in records if r["split"] == split))
            for split in ("dev", "selection", "test")
        },
        "median_words_by_split": {
            split: sorted(len(r["text"].split()) for r in records if r["split"] == split)[
                sum(1 for r in records if r["split"] == split) // 2
            ]
            for split in ("dev", "selection", "test")
        },
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (OUT / "split-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
