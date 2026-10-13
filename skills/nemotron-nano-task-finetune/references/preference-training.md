# Preference Training (DPO) After SFT

Treat preference training as an experiment with its own predeclared rule, not as the default next step
after SFT plateaus. In the case study, neither DPO arm passed its promotion rule; one pilot showed an
exploratory gain on a prompt variant, and the larger on-policy run lost substantially
([`case-study-atc.md`](case-study-atc.md)).

## Run the diagnostic first

For correctness-driven preference training, sample k answers per dev input (e.g. k=4 at moderate
temperature) and compare with greedy: greedy
accuracy, single-sample accuracy, pass@k, all-k-correct, and the share of inputs with both a correct and a
severe sample. pass@k − greedy is **oracle-selection headroom**, not recoverable greedy gain: it justifies
an experiment, not a forecast. For preferences among already-correct answers (style, brevity, tool
efficiency), measure the target dimension directly instead.

## Usually try something else first

- **Coverage gaps** (a category barely present in training): targeted SFT is the more direct fix.
  Verified chosen targets can carry new information, but DPO is a weak way to deliver it.
- **Rejected answers that are almost all confident errors.** One cheap way for the policy to lower their
  likelihood is to commit to less overall: answers get vaguer, hedge more, and drop details. Watch for
  this specifically (severe-error *types* shifting to omissions and hedges).
- **More targeted SFT on own errors:** repairs written candidate-blind and adjudicated against the frozen
  reference, through the normal data audit.
- **RL with a verifier** only where a subtask has a validated, sufficiently complete oracle, with
  reward-hacking tests (omission, hedging, parser exploits), coverage/abstention accounting, and an
  independent evaluator. Partial data-quality checks are **not** a reward for the whole task.

## If you run it

Declare in advance: policy and reference initialization (normally both the SFT checkpoint), beta, the
SFT-loss weight, LR, epochs/steps, pair counts, and the acceptance rule.

**Pairs**
- On-policy samples from the current model on mining-pool inputs only (never dev, reserve or test).
- Declare the preference dimensions for the task (correctness, severity, brevity, calibration, tool
  efficiency…) and a strength/confidence gate per pair. Chosen = a correct sample or a verified repair;
  both origins pass **one** clean-label gate. Rejected = a labelled worse answer on a declared dimension;
  a pair set made almost entirely of severe errors is the case most likely to teach vagueness. Report
  pair length and style biases. The labeller differs from the evaluator (the evaluator must not be the
  reward).
- Hygiene: enforce the admission gate; split the pair holdout by source group; one pair per unique input;
  keep template families on one side; re-label fallback chosen targets; avoid deterministic picks that
  bias wording (e.g. alphabetical); drop near-duplicate pairs with inconsistent labels; drop ties.

**Rendering**
- Render prompts with the **real SFT prep** so they match serving byte for byte (including the
  reasoning-mode prefix the task uses, e.g. an empty think block for no-reasoning Nano), and take the
  completion suffix from the prep output rather than hand-writing special tokens. Tokenize without
  adding extra special tokens.

**Loss balance**
- If the trainer's SFT term is a token-summed NLL, it can be one to two orders of magnitude larger than
  the preference term; scale its weight so neither term swamps the other, and check both in the smoke
  run. A weight that is too small may fail to hold specificity.

**Canary (all must pass before the real run)**
- Tokenize **both sides of every pair**: none dropped, truncated or with zero loss tokens.
- Masks cover only completion tokens.
- Finite gradients; the policy actually updates; the reference stays fixed.
- The initial *preference* component is ~0.693 (log 2) when policy = reference; step-0 preference
  accuracy can read 0 because ties count as not preferred.
- Export, reload and serve the canary checkpoint with the production request shape.
- Save the resolved configuration, actual pair counts and number of eligible steps.

**Reading results**
- A small drop in validation preference loss is weak evidence; judge the model on the same suites and
  rule as SFT.
- Infra: pin a trainer image known to work on your platform; give each run its own checkpoint storage
  (Nano checkpoints are large); pass the same storage settings to export as to training; serve the export
  with the SFT checkpoint's tokenizer and template.
