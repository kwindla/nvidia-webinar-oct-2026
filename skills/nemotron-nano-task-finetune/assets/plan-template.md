# <Task> Fine-Tune Plan

Working doc; keep it local. Every round appends a **dated declaration** before any of its outputs exist.

## 1. Goal and authority
- Task (user's words):
- Product and request shape (prompt versions, decoding, max tokens, reasoning on/off, history):
- Examples of current failures:
- Deadline / schedule / time reserved for owner validation:
- Budget cap and window (all items: teacher, critic, references, judge, baselines, training, exports, serving, storage):
- Version-control and repository rules:
- Pre-approved techniques:
- How failures may be fixed (model only / product guards allowed):
- Data rights confirmed (Y/N, by whom):
- Release authority (destination, artifact scope, audience, access) — or "not granted, ask before release":
- Sharing authority for grading pages / data:
- May the product's default model change? Who decides:
- Base model license and attribution:

## 2. Task type, contract and baseline
- Task type(s) (classification / extraction / open text / tool agent) and routing from task-types.md:
- Label sources (existing human labels, source spans, logs, teacher):
- Primary metric and evaluator per part (code / calibrated judge / trajectory):
- Reasoning mode used by the task (on/off) and decoding layer (constrained decoding, parser):
- Contract doc + version; severity definitions; absence semantics:
- Stress cases approved by owner (date):
- Frozen serving prompt(s) + version switch in the product:
- Baselines on the dev cohort (stock per prompt / simple baseline / frontier / prior model):
- Fine-tune justified? (gap on each declared goal: quality, latency, cost, footprint):

## 3. Decisions (from discovery questions)
- Single-purpose vs general; prompt formats trained; data audit owner; success margin; data cutoff:

## 4. Split (frozen <timestamp>, hash <...>)
- Grouping key and near-duplicate rule:
- Final test (size, stratum) — consumed? (date, lock commit):
- Selection / diagnostic / mining pool / reserve sizes (all allocated before mining? Y/N):
- Forced-to-dev rows; independence label per suite:

## 5. Data slices (per round)
| Slice | Inputs | Generator + version | Count | Kept | Rewritten | Dropped (by difficulty) |
|---|---|---|---|---|---|---|

## 6. Verification
- Deterministic checks (+ self-test count):
- Trusted labels: provenance, sampled audit size and disagreement handling (or n/a):
- Generated targets: critic family; blind reconstruction (separate call? Y/N); adjudication sources; human sample (or n/a):

## 7. Export contract
- Label (immutable) + manifest hash:
- Gates (quarantine, exact, slot-masked, lineage, checks, conflicting-ID failure, conditioning turns):
- Prep receipt: train/val files; records, conversations, target spans, packed sequences, total and loss tokens; packing/mask policy; drops explained:

## 8. Training
- Backend compatibility record (model revision, architecture, license, recipe, tokenizer/template,
  reasoning/tool mode, precision, sequence/packing, memory/topology, artifact types):
- Method (full SFT / PEFT: rank, alpha, dropout, targets; serve adapter or merged):
- Recipe + image + init checkpoint:
- Seq length (from measured p50/max):
- Comparison invariant (fixed updates / tokens / per-source example passes / measured compute); realized
  counts per arm from receipts and loader logs (examples, tokens, packs, per-source consumption):
- Seeds; save interval; checkpoints to export (declared):

## 9. Evaluation
- Suites + hashes:
- Evaluator identity per part: code scorer hash (schema, parser, metric config) and/or judge model/revision,
  prompt hash; full cache-key definition; judge calibration and held-out challenge results (or n/a):
- Report: fail-closed checks enabled; group-bootstrap CIs:

## 10. Final-test lock (fill before running; never edit after)
- Candidate(s) and comparators (weights paths + hashes / endpoints):
- Prompts and request settings:
- Verified cohort; disputed-reference rule:
- Evaluator identity and config hash (code scorer and/or judge):
- Analysis: metrics, intervals, success rule:
- Infrastructure-failure handling (retries; what counts as failure):
- Output paths; archive location:
- Lock commit (or hashed lock receipt if commits aren't authorized):
- Result and audit (reviewer, date):

## 11. Serving
- Deployment names (new per round), GPU, serving version and flags, experiment identity per arm:
- Known-good fallback live:

## 12. Cost ledger
| Item | Estimate | Actual |
|---|---|---|

## 13. Round declarations

### Round <N> — declared <YYYY-MM-DD HH:MM TZ>, before any outputs
- Evidence and hypothesis; lever:
- Arms / seeds / comparison type / updates:
- Selection procedure:
- Gates (all must pass):
- On pass / on fail:

### Round <N> — result <timestamp>
- Scores table (with CIs):
- Gate-by-gate outcome (exact numbers):
- Not-predeclared readings (labelled):
- Owner decisions (overrides recorded as such):
- Review file + findings folded in:
- Next lever:
