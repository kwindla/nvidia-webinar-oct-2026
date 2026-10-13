---
name: "nemotron-nano-task-finetune"
description: Orchestration skill for turning NVIDIA Nemotron 3 Nano 30B-A3B (or a sibling Nemotron LLM) into a narrow, single-purpose task model — for example the LLM turn of a voice agent — when prompting falls short. Use it to plan and gate a contract-first, measurement-first fine-tuning loop (baseline, frozen splits and rules, audited synthetic data, SFT through an existing recipe, multi-suite judging, predeclared decisions, quantization, release) and to answer whether DPO, more data, or a quantized build is justified. Execution goes to the user's stack or NVIDIA sub-skills.
triggers:
  - fine-tune Nemotron Nano for my task
  - fine-tune Nemotron 3 Nano
  - fine-tune an LLM for my voice agent
  - make a small model do one task well
  - specialize Nemotron for a domain
  - SFT Nemotron on synthetic data
  - teacher-generated fine-tuning data
  - my fine-tuned model hallucinates values
  - should I use DPO after SFT
  - quantize my fine-tuned Nemotron to NVFP4 or FP8
  - evaluate a fine-tuned LLM against stock
version: "0.3.1"
license: Apache-2.0
metadata:
  author: "Daily (derived from the ATC tutor project, nvidia-webinar-oct-2026)"
  tags:
    - nvidia
    - nemotron
    - nemotron-3-nano
    - llm
    - fine-tuning
    - sft
    - dpo
    - synthetic-data
    - evaluation
    - quantization
    - vllm
    - voice-agent
    - orchestration
  domain: ml
---

# Nemotron Nano Task Fine-Tune — Orchestration Skill

## What This Skill Is

A **high-level orchestration skill**, not a training manual. Given a goal such as *"make Nemotron 3 Nano
reliably extract and explain the key fields of a support ticket for our voice agent"*, it scopes the task into a
contract and a measured baseline, freezes the yardstick before any candidate output exists, builds audited
data, trains through a recipe that already works, judges candidates on several suites, decides by a
predeclared rule, iterates on measured failures, and — only when authorized — quantizes, releases and
validates in the live product.

It owns the plan, the gates and the decisions; the training/serving stack and sub-skills own execution.
The method is general; it was distilled from one project (an air traffic control tutor behind a voice
agent), recorded as a worked example in [`references/case-study-atc.md`](references/case-study-atc.md).
Domain rules, thresholds and commands from that project are examples, not requirements.

## When to Use

Use when someone wants a Nemotron LLM (default **Nemotron 3 Nano 30B-A3B**) to do one bounded task better
than prompting can, and can say what a correct answer looks like. Start here even if the user names a
technique (DPO, LoRA, NVFP4): treat it as a candidate until the loop shows it is needed.

Do **not** use for open-ended capability gains with no task contract, fast-changing knowledge (use
retrieval), or a problem the measured prompted baseline already solves. If no GPU budget or training
stack exists, stop after Stage 3 with the plan and the baseline.

## Orchestration Workflow

Announce each step (**Step N/15 — Title**). Detail: [`references/workflow.md`](references/workflow.md).

| # | Stage | What happens |
|---|---|---|
| 1 | **Goal, authority, budget** | Task in the user's words; spend cap and window; what may run unattended; release and sharing permissions (separate from permission to train) |
| 2 | **Task type, contract, baseline** | Route the task type (classification, extraction, open text, agent); contract and severity; owner-approved stress cases; frozen prompt and decoding layer; measure prompted stock, a simple baseline, a frontier model |
| 3 | **Plan + adversarial review** | Plan doc with a dated acceptance rule; second-agent review; revise; re-review |
| 4 | **Split and freeze** | Group by source; allocate **every** holdout now (final test, selection, diagnostic, reserve) before any mining; hash everything |
| 5 | **Canary** | ~100 rows and every conversation format through the real prep → short train → export → serve → one request per arm |
| 6 | **Assemble labels** | Trusted labels first; teacher targets only where none exist; code supplies values; provenance-bound, ledgered |
| 7 | **Audit labels** | Trusted labels: provenance + deterministic and sampled checks. Generated semantic targets: checks → independent critic → evidence-based adjudication → drop on dispute |
| 8 | **Gated export** | Immutable label; leakage gates over every turn and lineage; prep receipts |
| 9 | **Train** | Tested recipe; seq length from measured lengths; a declared comparison invariant (updates, tokens, or per-source example passes) verified from receipts |
| 10 | **Export and serve candidates** | Export only predeclared checkpoints; serve side by side with the product's request shape |
| 11 | **Evaluate every suite** | Task-type metrics and suites (code where possible, calibrated judge where not); fail-closed report |
| 12 | **Decide, review, iterate** | Apply the declaration as written; review; next lever from error analysis → back to 6 |
| 13 | **Final test (once)** | Lock candidate, comparators, request, cohort, analysis; run once; audit; archive. Never re-judged after |
| 14 | **Quantize (optional)** | Supported native export or known recipe, pinned tools, in-distribution calibration; layout check only where a published build exists; accept vs same-stack BF16 |
| 15 | **Release and live-validate** | Only with explicit authorization, at the authorized visibility (private if unspecified): verified upload, reader check; live smoke; owner validation |

Stages 6–12 are the **inner loop**. Stage 13 needs an untouched test; once consumed, later claims are
development-only unless fresh held-out data is collected.

## Sub-Skills and Stacks

Registry and handoffs: [`references/sub-skills.md`](references/sub-skills.md).

| Role | Prefer | Alternative |
|---|---|---|
| Training (SFT, PEFT, DPO) | The user's **existing, tested** recipe | `nemotron-customize`; `launch-nemo-rl` for NeMo-RL on Kubernetes |
| Synthetic data | Project scripts calling a strong teacher with schema-constrained JSON | `data-designer` |
| Quantization | The recipe used for previous in-house quants | `nemotron-customize` ModelOpt steps |
| Serving | vLLM with Nemotron flags ([`references/training-serving.md`](references/training-serving.md)) | NIM |
| Adversarial review | A second agent from a different model family | A fresh-context subagent, disclosed as same-family |

Check what is installed before naming a sub-skill; if one is missing, say so and continue with the
interim guidance in the references.

## Key Principles

- **Declare the rule before the output exists**, with a timestamp; apply it as written; a near-miss is a
  fail. Readings outside the rule are labelled "not predeclared." An owner may override; record that as an
  owner decision, not a pass.
- **Final evidence is immutable.** The final test runs once against a complete lock and is never
  re-frozen, re-judged or trained on. Development evaluation is versioned separately.
- **Allocate holdouts before mining.** A set carved out after its relatives trained a model is not
  held out; label legacy exposure honestly.
- **Fail closed on evidence, for every task type.** Scores — from code or a judge — bind to every input
  and evaluator version that produced them; duplicate, stale, partial or mixed cohorts stop the report.
- **Measure noise before chasing gains**: seed spread, group-bootstrapped paired CIs, serving-stack drift.
  Replicate in proportion to the stakes; a declared single-seed exploratory run is allowed.
- **Compare like with like**: declare the invariant first (fixed updates, fixed processed tokens, fixed
  per-source example passes, or measured compute), name the comparison after it, and verify the realized
  counts from receipts; compare quantized builds with BF16 on the same serving stack.
- **Route by task type.** Classification and extraction use trusted labels and code metrics; LLM judges
  are for open-text criteria without an exact oracle; agents are judged on trajectories.
- **Code decides only what code can decide.** Partial checks gate data and diagnose outputs; they are not
  correctness oracles and not RL rewards for the whole task.
- **Labels come from what the model can see**, and every row is bound to the contract, facts, models and
  checkers that produced it.
- **Fix failures where the owner wants them fixed.** If the goal is the model itself, fix errors with
  data, checkpoint choice or training, not output filters. If the owner wants a product guard, add it,
  but measure the unguarded model separately and don't credit the guard to the model.
- **Start from the tested recipe**; declare and canary any method or model change. Promote checkpoints
  on task metrics, not trainer loss.
- **Every pipeline step stops on failure** and is verified by its own evidence. Act on completions.
- **Owner knowledge outranks the teacher**; corrections propagate forward to targets, development
  references and judge — never into frozen final evidence.
- **Nothing leaves the machine without authorization**: uploads, shared grading pages, changing the live
  default, pushes.
- **Review at every boundary**; fold in fixes; ask for a follow-up verdict.

## Source of Truth

| Topic | Location |
|---|---|
| Stage detail | [`references/workflow.md`](references/workflow.md) |
| Task-type routing (labels, metrics, judges, suites) | [`references/task-types.md`](references/task-types.md) |
| Data generation, audit, export | [`references/data.md`](references/data.md) |
| Suites, references, judges, reports, decision rules | [`references/evaluation.md`](references/evaluation.md) |
| Nano training, serving, quantization, release settings | [`references/training-serving.md`](references/training-serving.md) |
| DPO after SFT | [`references/preference-training.md`](references/preference-training.md) |
| Worked example (ATC tutor) | [`references/case-study-atc.md`](references/case-study-atc.md) |
| Plan/declaration template | [`assets/plan-template.md`](assets/plan-template.md) |
| Nemotron 3 Nano model card and license | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| NVIDIA skills catalog | https://github.com/NVIDIA/skills |

## Limitations

- Orchestration only; execution belongs to the user's stack or the NVIDIA sub-skills.
- Hyperparameters and thresholds are starting points from one project, not rules. Prices are historical
  estimates.
- Needs training GPUs (full SFT of Nano used one 8×B200 node). Teacher, critic or judge API budget is
  needed only where the task routes to generated labels or LLM judging.
- Flags, presets and image tags drift across NeMo, ModelOpt and vLLM releases; verify before running.
- The evals in `evals/` are qualitative prompts; they have not been run in a harness.
