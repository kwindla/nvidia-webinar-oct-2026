# Evaluation: Suites, References, Judges, Reports, Decisions

Build and freeze the evaluation stack **before** candidate outputs exist; it is the asset most worth
protecting. Choose metrics and evaluators by task type ([`task-types.md`](task-types.md)): code metrics
wherever an exact oracle exists, a calibrated LLM judge only for open-text criteria without one, and
trajectory metrics for agents. Section 3 (references) and the judge-validation part of section 4 apply
to LLM-judged parts; the evidence-integrity rules in section 4 apply to **every** task, including
code-scored ones. Example sizes and outcomes from one project: [`case-study-atc.md`](case-study-atc.md).

## 1. Suites

Use the suites that fit the task; capability, robustness and mode suites are optional by type.

| Suite | Typical size | Built from | Purpose | Never |
|---|---|---|---|---|
| **Final test** | ~20% of source groups | held-out groups, stratified | one locked run for the headline claim (workflow Stage 13) | rerun, re-freeze, re-judge, train, select |
| **Selection** | ~10% of dev | dev, excluded from generation | early prompt/checkpoint choices | mine |
| **Dev cohort** | a few hundred | never-mined dev | checkpoint development each round | mine |
| **Mining pool** | the rest of dev | dev | errors for repairs or preference pairs; train-side inputs | report as held out |
| **Reserve** | a few hundred | groups withheld at split time | one-time promotion check once dev is adaptively exposed | reuse |
| **Capability suite** | ~100+ | code templates, separate seed, exact-text excluded from training | deterministic pass/fail per category | judge with an LLM |
| **Robustness suite** | ~80 | corruptions of held-out capability items | inventions / faithfulness plus a deterministic lost-value check | trust alone |
| **Mode suite** | ~100+ items in families | written fresh, n-gram-filtered against every eval input | task vs chat vs question handling, multi-turn, factual answers | — |
| **Judge challenge set** | correct answers + one-change near-misses | train-side inputs | false rejects / accepted near-misses | tune the judge on its held-out half |
| **Owner calibration** | 30–40 | stratified sample | human agreement gate for the judge | show the judge's verdict before grading |
| **Live smoke** | ~20 turns | real upstream path | latency + qualitative | — |
| **Owner validation** | ~20 probes | the live product | final gate | — |

Label each suite's independence honestly: exact-text exclusion, lineage exclusion, template- or
composition-disjoint, or genuinely new sources. Template-built capability and robustness suites share
structure with template-built training data; they mostly measure interpolation, and the robustness
suite usually resembles the training noise. The live product is the real robustness test.

## 2. Metrics

- **Primary metric by task type** (task-types.md). For open text, "correct" = every required element
  conveyed, nothing unsupported asserted, role or attribution acceptable. Timeouts, truncations and
  invalid structured output count as failures; an empty answer is wrong unless the contract makes
  "nothing" the correct output.
- **Severe rate**: any error class the contract marks unacceptable (wrong value or unit, changed
  instruction, lost negation or condition, omitted action, invented entity, mode error, unlicensed
  guess…). Break it out by type.
- **Diagnostics** (unsupported-claim rate, missing-required, form/length, "correct apart from X") never
  replace the primary metric mid-project.
- **Intervals**: paired bootstrap 95% CIs resampled by **source group**, not by row.
- **Baselines in every report**: stock with the best dev-selected prompt (the bar), stock with other
  prompts, the task's simple baseline, a frontier model when affordable (context), the incumbent.

## 3. References

- Written **candidate-blind** from the gold input by a strong model, with structured fields: required
  and optional elements, acceptable alternative readings, role/attribution, ambiguity class, licensed
  readings, prohibited completions.
- A separate **verify** call may correct *any* field (max two rounds); still-disputed rows are excluded
  and counted. Expect it to revise a large share of first drafts.
- **Hash-frozen.** A contract change creates a **new development reference version** (targeted or full
  recheck) and every development arm is re-judged under it. The final test's references and judgments are
  never re-frozen or re-judged.

## 4. Judges and evidence integrity

- Primary judge: a strong model, **blinded to system identity**, schema JSON output.
- **Experiment identity** (fixed per arm): source and request hashes; model or adapter revision; for
  self-hosted weights, the path and hash, tokenizer and chat-template revision, serving image and flags,
  GPU; for hosted models, the requested and returned model name/version and whatever attestation the
  provider offers — mark unavailable fields as unavailable, never invent them.
- **Row outcome** (varies per response): answer, request/response IDs, attempts, error, finish reason,
  timing. Failures are scored by the declared rule (usually as wrong), not dropped.
- **Evaluator identity:** for code scorers, the hash of the scorer, schema, parser/normalization and
  metric configuration; for judges, the items below. Cached scores bind to it either way.
- **Cache key = hash of every input that affects the verdict**: the actual input and conversation
  history as rendered for the judge, the candidate answer, the row outcome state that affects scoring
  (finish reason, error, truncation), the reference, judge model and revision, judge prompt,
  schema/tool configuration, decoding and reasoning settings, contract and table versions. Store
  verdicts by that key, separately from row identity; reattach current metadata at report time.
- **Reports fail closed.** Before computing anything, require: unique IDs (duplicates are an error, not
  silently collapsed); every arm complete relative to the **source cohort**, not merely equal to each
  other; fixed, declared exclusions; every verdict's key recomputed from current outputs and inputs and
  matching; one experiment identity per arm. Any failure stops the report.
- **Validate the judge** before relying on it:
  - test-retest on the full set — this measures repeatability, not bias;
  - human calibration with a **predeclared disagreement gate** (e.g. ≤6 of 30), blind grading first;
  - a near-miss challenge set split into a tuning half and a **held-out half** that is never used to
    adjust the judge prompt or rubric; report false accepts and false rejects with counts and intervals;
  - after a material judge change, re-audit fresh disagreements.
- If the judge shares a family with the teacher, disclose it. A cross-family judge is worth trying, but
  measure its agreement before using it for decisions.

## 5. Human in the loop

- A grading page showing input, answer and a verdict box, with the judge's verdict revealed only after
  the human grades. Sharing such a page outside the machine needs authorization (workflow Stage 1).
- Turn the owner's style and domain notes into contract rules. Don't keep calibrating a judge that is
  about to change; calibrate the new one on items that stress the new rules.

## 6. Decision rules — declare before outputs

| Decision | Rule shape |
|---|---|
| First fine-tune vs stock (final test) | tuned ≥ best dev-selected stock + margin, and unsupported/severe ≤ stock (point rule; CIs reported) |
| New data round vs incumbent | dev ≥ incumbent − tolerance; severe ≤ incumbent or a ceiling; capability ≥ floor; mode errors ≤ allowance; robustness improves; factual errors ≤ ceiling |
| Checkpoint choice | best seed-mean checkpoint → better seed there → seed veto → ties to lower severe |
| Schedule study | "better" if mean ≥ +δ and most seeds beat; "worse" if mean ≤ −δ |
| Reserve promotion | +margin, severe no higher, mode errors no more, owner validation |
| Preference stage | dev +δ, severe no higher, capability no fewer, mode no more, factual/style no worse |
| Quantized build | vs **same-stack BF16**: dev ≥ −δ, severe ≤ +δ, capability ≥ −k, mode ≤ allowance, robustness ≤ +k |
| Ship | owner validation agrees |

Set δ and margins from measured noise: seed spread on this task, paired CIs, and how much the serving
stack alone moves scores. Report paired uncertainty rather than treating one historical standard
deviation as a significance threshold.

## 7. Wording results

Disclose: which numbers are development-only and how dev was reused; that a *configuration* (data,
prompt, schedule, selection) was chosen, not isolated weights; the exact scored denominator and
exclusions; judge/teacher family overlap; where the frontier baseline is better. Distinguish "failed its
predeclared promotion rule" from "does not work." Keep exactly one headline claim per consumed final test.
