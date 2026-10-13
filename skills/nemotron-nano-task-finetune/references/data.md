# Data: Generate, Audit, Export

Training data for a narrow task model combines trusted existing labels (human labels, source spans,
logged episodes) with teacher-written targets where none exist. Route the label source by task type
first ([`task-types.md`](task-types.md)). The bar: a reviewer cannot find a target, or a conditioning
turn, that teaches a wrong habit.
Concrete numbers from one project are in [`case-study-atc.md`](case-study-atc.md).

## 1. Inputs: what the model will actually see

- **Size from learning curves and slice deficits**, not a fixed row count: train on growing subsets or
  read per-slice errors, and add data where errors concentrate. Enumerating what must be learned
  (vocabulary, value types, compositions) is a useful first estimate.
- **Existing labels are an asset.** Keep approved labels; audit a sample of them; let a teacher propose
  labels only for unlabeled rows, and never silently replace a human label with a synthetic one.
- **Prefer real inputs where they exist.** Clean synthetic templates teach the textbook case; real
  inputs carry the length, disfluency and mess the product sees. Measure performance by input length or
  difficulty to see the gap.
- **Keep a clean core** for canonical cases. Existing-label tasks can use trusted, reviewed examples.
  For synthetic canonical targets, use a code-built slice whose writer and critic both see a
  **facts block** (the exact values and their correct renderings). Cleanliness filters on real data can
  silently remove whole categories; check coverage after filtering.
- **Variants** (value swaps, paraphrases) of real parents when the teacher judges them safe; they
  inherit the parent's split.
- **Robustness inputs** if the model sits behind a noisy upstream system (ASR, OCR, retrieval):
  realistic corruptions of train-side inputs (substitutions, drops/merges, fragments), or the real
  upstream system run on synthetic input. Check that the corruptions resemble live errors.
- **Mode inputs, if the product mixes modes:** talk *to* the assistant (greetings, meta questions)
  alongside task inputs, plus **minimal pairs** where the same words appear in both roles. Robustness
  data built from real inputs can otherwise teach the model to treat ordinary chat as task input. A
  pure classifier or extractor needs out-of-scope inputs instead, not chat.
- **Agent episodes:** keep tool calls and results in the trainer's native format (task-types.md).
- **Session-start rows** if the product speaks first.

## 2. Teacher-written targets (where trusted labels don't exist)

- A strong teacher with **schema-constrained JSON** (strict `additionalProperties: false`), streaming,
  retries.
- **Write from the input, never by polishing an old target.** Old targets carry old conventions.
- **Values and entities from code**: typed tables sourced from the domain's authoritative documents;
  checked entity tables. Don't rely on teacher or judge recall for names, codes or conversions.
- **Labels from visible input only.** For degraded inputs, the target says what the input supports and
  marks uncertainty; it never restores a value the input lost.
- **Provenance binding.** Every row and every audit verdict records the hashes or versions of: input,
  parent row, contract, prompt, facts and entity tables, writer model and settings, critic model,
  checker code. Resume and caching key on that whole tuple, not on row ID. When any element changes,
  regenerate or re-audit. Conflicting rows with the same ID fail the export instead of one being picked.
- **Resumable and append-only**, with a **cost ledger** (tokens and dollars per call). Batch APIs are
  cheaper; direct concurrent calls are faster.
- Defensive API handling: refusals can be triggered by innocuous schema field names (bisect the prompt);
  some schema keywords may be unsupported (enforce limits in code); structured items can come back as
  encoded strings (parse defensively); credit or rate exhaustion mid-run (wait and resume).
- A **writer bake-off** (two candidate teachers on ~60 inputs, a blind pairwise page for the owner) is
  cheap and settles which teacher writes.

## 3. Audit

**Trusted labels** (approved human labels, source spans, logged episodes): record provenance; run the
deterministic checks below; audit a stratified sample with domain review; send disagreements to the
owner. A model's disagreement never replaces a human label on its own. No LLM critic is needed.

**Generated semantic targets:** code → critic → adjudication → human, as follows.

1. **Deterministic checks** decide only what code can decide: form rules from the product contract,
   value rendering and unit conversions, polarity and condition markers, plausible ranges for each value
   type, leakage. Everything else is `needs_review`, never a pass. These checks are partial by design:
   use them to reject and diagnose, not as a correctness oracle or a reward for the whole task.
2. **Self-test the checks** with deliberately wrong outputs that must fail and valid paraphrases that
   must pass; add a case each time a check misfires.
3. **Critic** from a different model family than the writer. For a candidate-blind claim, reconstruct the
   meaning in a **separate call that does not see the candidate**, then compare in a second call.
   Verdict pass / fix / drop; a fix is new data and re-passes the checks; at most two repairs, then drop.
4. **Adjudication** for disputes: give the adjudicator the retrieved text of the authoritative source,
   not only a request to cite from memory. Unresolved rows are dropped.
5. **Re-audit after transformations.** Restyles and repairs are new data.
6. **Audit every conditioning turn**, not only trained targets: system prompts and untrained assistant
   history in multi-turn rows also teach.
7. **Human:** a small blind-graded sample by the owner. Their notes often change the contract. Inspect
   the first pilot's critic rewrites before generating at scale; a high rewrite rate means the writer
   prompt or contract needs work.

Report keep / rewrite / drop rates **by slice and difficulty**, so filtering does not quietly remove the
hard cases. Treat the rates as diagnostics, not targets.

## 4. Export with gates

Use the trainer's native format with a per-turn training mask (tool-call and tool-result messages
included for agents), for example:

```json
{"conversation_id": "...", "messages": [
  {"role": "system", "content": "...", "train": false},
  {"role": "user", "content": "...", "train": false},
  {"role": "assistant", "content": "...", "train": true}]}
```

- Train only on target assistant turns.
- **Prompt formats follow the product.** If the product can send several system prompts (or none),
  assign one format per row by a stable hash, so each row is rendered once.
- **Gates on the source-bearing content of every turn**, including untrained history:
  - the source group must be train-side;
  - no exact or task-appropriate near-duplicate of any held-out item's source content (audit borderline
    matches by hand rather than masking so broadly that distinct examples merge);
  - lineage: every descendant (variant, corruption, repair) follows its seed's split;
  - checks re-run on every trained turn;
  - identical duplicate rows dedupe; conflicting rows fail.
  Reusable content is not leakage: system prompts, tool schemas, output schemas, generic requests and
  label names legitimately appear in every split. Exact-string gates alone miss template leakage, and
  exempting chat or descendants is the usual hole.
- **Validation split by seed family** (a seed with all its descendants), plus a few rare compositions;
  keep chat and mode families on one side. Some trainers copy validation to a file named `test.jsonl`;
  that is **not** the final test — label it.
- **Immutable labels** with manifest hashes; never rewrite a label.
- **Prep receipt** per label: named train and validation files; counts of source records,
  conversations, trained target spans, packed sequences, total and loss-bearing tokens; the packing,
  reset and mask policy; sampling weights; a loss-mask check; the token-length distribution; and every
  dropped or truncated row explained (zero unexplained). The schedule's unit count comes from this
  receipt and the trainer's resolved config.
