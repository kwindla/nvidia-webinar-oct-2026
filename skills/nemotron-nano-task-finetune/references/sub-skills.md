# Sub-Skills and Stacks

This skill plans, gates and decides. Execution goes to whatever the user already runs, or to NVIDIA
catalog skills. Check what is installed (`npx skills@latest list`) before naming a sub-skill as
available. Installing a skill is a change to the user's environment: ask first
(`npx skills@latest add NVIDIA/skills --skill <name>`).

## Training (SFT, PEFT, DPO, RL)

- **Prefer the user's existing, tested recipe** as the baseline. Ask for it in Stage 1; declare and
  canary any method or model change.
- **`nemotron-customize`** (NVIDIA): plans and configures repo-native Nemotron steps — SFT/PEFT via
  AutoModel or Megatron-Bridge, DPO/RLVR/GRPO, checkpoint conversion, ModelOpt optimization, evaluation of
  checkpoints or endpoints. Hand it the base checkpoint, export path and format, sequence length,
  updates or epochs, seeds, save interval, hardware, and the PEFT settings if any. It works from a
  Nemotron repo checkout; route through its catalog, artifact and hardware references and verify the
  selected live recipe (artifact types, tokenizer, packing, sequence length, topology) before running.
- **`launch-nemo-rl`** (NVIDIA): launching and debugging NeMo-RL recipes on Kubernetes.

## Data

- Trusted labels where they exist; otherwise project scripts calling a strong teacher with
  schema-constrained JSON. The project's own checks and exporter in both cases; a critic only for
  generated semantic targets (data.md).
- **`data-designer`** (NVIDIA): synthetic text generation for inputs or target drafts. Its output passes
  the same audit and export gates.

## Evaluation

Project-owned: code scorers (schema, parser, normalization, metric config) frozen and hashed; for
open-text parts, frozen references and a blinded judge; full-input cache keys and a fail-closed report for
every task type (evaluation.md). Catalog tools can run standard benchmarks on an endpoint; task-contract
scoring stays with the project.

## Quantization

A supported native export or a known in-house recipe; otherwise `nemotron-customize` ModelOpt steps.
Check the layout against NVIDIA's published build only where one exists for the same model and format;
accept on task quality against same-stack BF16.

## Serving and product

vLLM or NIM. The product should select both the model endpoint and its prompt version by configuration,
so a candidate can be swapped in and rolled back.

## Review

A second agent from a different model family with read-only access, asked to write findings with
severities to a review file and to signal when done. Fold in, then ask for a follow-up verdict on each
finding. If no second agent exists, use a fresh-context subagent with an explicitly adversarial brief
and disclose that the review is same-family. If neither is available, say so, do a structured self-review
against the review checklist, and offer an external review handoff — never report that an independent
review ran when it didn't.
