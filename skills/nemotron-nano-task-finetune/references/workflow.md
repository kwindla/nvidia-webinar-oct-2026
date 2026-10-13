# Orchestration Workflow

The end-to-end loop for "fine-tune a Nemotron LLM to do this one task well." Announce each step
(**Step N/15 — Title**). Stages 6–12 repeat once per data round; each round is a dated declaration in the
plan doc ([`../assets/plan-template.md`](../assets/plan-template.md)).

Task routing: [`task-types.md`](task-types.md). Data: [`data.md`](data.md). Evaluation:
[`evaluation.md`](evaluation.md). Backend and stack: [`training-serving.md`](training-serving.md). A worked example of every stage:
[`case-study-atc.md`](case-study-atc.md).

---

## 1. Goal, authority, budget — *Orchestration*

Capture in the user's words:

- **Task:** input, output, the product or system it serves, and real examples of current failures.
- **Deadline and schedule**, including time reserved at the end for owner validation. A new round is
  admissible only if it can finish (train, judge, review) with that reserve intact.
- **Budget:** one total cap and time window covering every cost: teacher, critic, reference and judge
  calls; frontier baselines; training; exports; concurrent serving; storage. Get current prices before
  quoting. Track committed and running jobs, not just finished ones. Ask before exceeding the cap or
  after the window closes.
- **Authority for unattended work:** version-control rules, which repositories and branches may be
  edited, which techniques are pre-approved, whether data may be sent to remote training and to
  teacher/judge APIs.
- **Release and sharing authority, separately:** may weights, data, or evaluation pages leave the
  machine — to which destination, audience and access level? May the product's default model change?
  Permission to train implies none of these. Record the base model's license and attribution terms.
- **Owner constraints on how failures may be fixed** (e.g. "fix it in the weights, no output filters",
  or "a production guard is fine"). Follow the current owner's choice.
- **Data rights:** confirm the user may train on the data; once confirmed, don't re-raise it.

## 2. Contract and baseline — *Orchestration + owner*

1. **Read the product.** The prompt it sends, decoding settings, max tokens, reasoning on/off,
   conversation history, output format. The fine-tune must serve that request shape.
2. **Classify the task** (classification, extraction/structured output, open text, tool agent, or a
   mix) and route labels, metrics, judges and suites with [`task-types.md`](task-types.md).
3. **Profile the data:** counts, lengths, existing labels and their provenance, coverage per category.
   Name thin categories now.
4. **Write the behavior contract** for the task type, plus **severity definitions** (which errors are
   unacceptable). For open text a typical priority order is: content correct → invent nothing →
   role/attribution → stated uncertainty → form and length → entity rendering → off-task input. For
   machine-consumed output, schema validity is a gate, not a low-priority style rule.
5. **Stress cases:** ~10 hard inputs with gold answers, approved by the owner before generation.
6. **Freeze the serving prompt(s)** and any decoding layer (constrained decoding, parser). A later contract change means a new prompt version and a new judge
   version; the product selects the prompt by version.
7. **Measure baselines** on the dev cohort with the product's settings: the stock model with each
   candidate prompt (the best dev-selected one is the bar), the task's simple baseline, a frontier model
   when affordable (context), any in-house prior model. If prompting meets **every** declared goal
   (quality, latency, cost, footprint), stop: no fine-tune.
8. **Ask questions until the plan is decidable** (single-purpose vs general model, prompt formats, who
   audits data, required margin over stock, data cutoff). Record answers.

## 3. Plan and adversarial review — *Orchestration + reviewer*

Write the plan doc with: Decisions, Split, Contract, Data slices, Verification, Export contract,
Training, Evaluation, Final-test lock, Serving, Cost, Gates, and the first dated acceptance rule. Send it
to a second agent for an adversarial review (blocker/high/medium findings written to a file); fold in;
revise; ask for a follow-up verdict, or a from-scratch review when the user asks.

## 4. Split and freeze — *Data*

- Group rows by the **deployment generalization unit** (recording, document, session, customer,
  episode, or a time window when deployment sees the future), then union groups that share
  source-bearing content (exact or task-appropriate near-duplicates). Shared instructions, schemas,
  tool definitions and label names are expected across splits and are not leakage.
- Force already-exposed rows (prompt examples, stress cases, anything inspected) to the dev side.
- **Allocate every holdout now, before any mining or training:** a final test (~20% of groups,
  stratified on a natural stratum), a selection slice, a never-mined diagnostic set, and a one-time
  promotion reserve if dev will be reused heavily. What remains is the mining pool.
- If a holdout must be carved out later, audit each comparator's exposure to its ancestors; for an
  exclusion claim, remove every descendant and retrain from an unexposed initialization. Otherwise label
  the set "newly withheld", not pristine.
- Hash the split, references, judge config and suites. Never change a split after seeing scores.

## 5. Canary — *Training stack*

Push ~100 pilot rows and every conversation format the product uses (each prompt variant, rows without
a system prompt, assistant-first turns, multi-turn) through the **real** prep → a few-step train → export
→ an isolated endpoint with the product's request shape → one request per benchmark arm. Use separate
canary labels for data, prep, run and serving. Every new trainer (e.g. DPO) gets its own canary
([`preference-training.md`](preference-training.md)).

## 6. Assemble labels and targets — *Data*

Start from trusted labels where they exist (approved human labels, source spans, logged episodes); have
a teacher write targets only where they don't. Teacher targets are written from the input, never by
polishing an old target; values and entities come from code; everything is provenance-bound, resumable
and cost-ledgered. Choose slices to fix measured failures. [`data.md`](data.md) §1–2.

## 7. Audit labels — *Data*

Two branches ([`data.md`](data.md) §3):

- **Trusted labels** (approved human labels, source spans, logged episodes): record provenance, run the
  deterministic checks, and audit a sample with domain review. Disagreements go to the owner; a model's
  disagreement never overwrites a human label on its own. No LLM critic is required.
- **Generated semantic targets:** deterministic checks → independent critic with a candidate-blind
  reconstruction → ≤2 repairs → adjudication against retrieved authoritative sources → drop what stays
  disputed. Inspect the first pilot's rewrites before scaling; re-audit after any transformation.

Sending data to a teacher, critic or judge API needs the data-transfer authorization from Stage 1.

## 8. Gated export — *Data*

Immutable label; train flags only on target turns; leakage gates over every turn (including untrained
history) and every descendant; prep receipts with counts. [`data.md`](data.md) §4.

## 9. Train — *Training stack*

- Start from the user's tested recipe as the baseline. Data-driven changes (sequence length from
  measured token lengths, updates, save interval) need no special justification; method or model
  changes (full SFT ↔ PEFT, another Nemotron size, longer context) are allowed when declared and
  justified, and get their own canary. Run the backend compatibility check in
  [`training-serving.md`](training-serving.md) §0 first.
- **Declare the comparison invariant** for each round, and name the comparison after it:
  - *fixed updates:* same optimizer updates and LR horizon; with unchanged uniform sampling, a larger
    corpus then gives each example fewer passes;
  - *fixed tokens:* same processed (or loss-bearing) tokens — a token budget, not equal passes over any
    particular slice;
  - *fixed exposure:* same passes per supervised example for a named source or slice;
  - *fixed compute:* only when compute is actually measured.
  Get update counts from the trainer's resolved config, and realized counts (examples, tokens, packed
  sequences, per-source consumption, repetition and truncation) from the prep receipt and loader logs.
  Equal pack or batch counts do not imply equal example exposure when packing density differs. Verify
  the logged `train_iters` before waiting on it.
- Replicate in proportion to the stakes; a single-seed run is acceptable when declared exploratory or
  when it retrains an already-measured corpus.
- Save several checkpoints; declare in advance which will be exported. Promote on task metrics; a
  representative held-out loss may be secondary evidence, but never small-batch trainer validation loss.
- Run each round as one chain (export → deploy → generate answers → judge → report) where every step
  stops on failure and asserts its own success marker.

## 10. Export and serve candidates — *Serving stack*

- Export only after the training job has fully exited (some storage commits only at exit).
- Unique, never-overwritten names per checkpoint; never export over a path being served.
- Serve candidates side by side under a new deployment name per round so the incumbent stays up, or
  sequentially when capacity requires it. Retire candidates once the round is decided; keep the
  fallback. Warm endpoints before measuring.
- Use the product's request shape. Record each arm's experiment identity (evaluation.md §4).

## 11. Judge every suite — *Evaluation*

Every suite for every candidate and baseline; the report fails closed on incomplete, duplicate, stale or
mixed evidence. [`evaluation.md`](evaluation.md).

## 12. Decide, review, iterate — *Orchestration + reviewer*

- Apply the declaration exactly (e.g. best seed-mean checkpoint → better seed at that point → seed veto
  → all gates). Record the outcome, including fails by fractions of a point.
- Ask the reviewer to check the result and the next round's design; use them as an advisor before costly
  decisions.
- Pick the next lever from **error analysis**. Common patterns (hypotheses to test, not laws):

  | Evidence | Lever to try |
  |---|---|
  | Leakage found | rebuild with full gates under a new label; demote the old model to exploratory |
  | One error category dominates | targeted slice with code-checked values and a facts block |
  | Real, messy inputs score far below clean ones | real train-side inputs rather than more synthetic templates |
  | Owner style or domain corrections | contract revision → new *development* references → re-judge development arms |
  | Errors on noisy upstream input (ASR, OCR, retrieval) | robustness slice labelled from visible input, plus a lost-value check |
  | Robustness data confuses ordinary inputs | minimal-pair contrast data (same words, two roles) |
  | Larger corpus at fixed compute lost dev | a fixed-exposure retrain as the controlled test |
  | pass@k ≫ greedy | consider preference training — read [`preference-training.md`](preference-training.md) |
  | All runs peak at the same point | a controlled schedule study before drawing conclusions |

- Keep a known-good fallback live until a candidate passes and the owner agrees.

## 13. Final test (once) — *Orchestration + reviewer*

Only if an untouched final test exists.

1. **Lock** before any output: candidate(s) and comparators (exact weights or endpoints), prompts and
   request settings, the verified cohort and the rule for disputed references, judge identity and
   config, the analysis (metrics, intervals, success rule), the handling of infrastructure failures
   (retry policy, what counts as a failure), and output paths. Commit the lock, or write an
   immutable hashed lock receipt if commits aren't authorized.
2. **Run once**; archive complete raw outputs, judgments and the report.
3. **Audit** with the reviewer: complete unique ID sets per arm, hashes match the lock, request settings
   match, rule applied correctly.
4. The test is now **consumed**: never rerun, re-frozen, re-judged or trained on. A later contract change
   creates a new development evaluation version only. Another confirmatory claim needs fresh, untouched
   data. Label every later result development-only.

The final test can run whenever a candidate is worth a headline claim — early (to establish that
fine-tuning works at all) or at the end. It can run only once.

## 14. Quantize (optional) — *Training stack / `nemotron-customize`*

Prefer a supported native export path; reuse a known quantization recipe if one exists (pin its image
and tool versions). Calibrate on in-distribution training rows. If the vendor publishes a quantized build
of the same model and format, check the export's layout against it; otherwise rely on the native export
and a serving load test. Declare the rule first and compare task quality against **BF16 on the same
serving stack**. Pick the format by target hardware. [`training-serving.md`](training-serving.md) §4.

## 15. Release and live-validate — *Orchestration + owner*

- **Release only with explicit authorization** from Stage 1 (destination, artifact scope, access). If it
  wasn't granted, prepare and validate the release, then ask immediately before uploading. Don't re-ask
  for approvals already given. The same applies to shared grading pages and changing the product default.
- Release steps: the recorded visibility (private if unspecified) → assert the access level → upload from where the weights live →
  verify each remote file by size → load-check the uploaded files → a model card with the trained system
  prompt, serving command, client example, hardware note, scores with caveats, and the base model's
  license and attribution.
- Check access with the intended reader's credentials.
- **Live smoke:** drive the real upstream path with the product's settings and accumulated history for
  ~20 turns; report latency.
- **Owner validation is the final gate:** give the owner the live product and ~20 probes (solid cases,
  known weak spots, off-task and mode probes); afterwards separate model failures from pipeline failures.
  Findings start the next round.
- Update the product's documentation and configuration; keep the previous model as a commented fallback.
