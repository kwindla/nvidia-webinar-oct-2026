# Case Study: the ATC Tutor (Nemotron 3 Nano 30B-A3B)

One application of this skill, recorded so a reader can see the method in practice. Everything here is
specific to that project; the general guidance is in the other references. Sources are in the
`nvidia-webinar-oct-2026` repository: `docs/atc-tutor-*.md` (plan, contracts, evaluation, results,
analysis) and `docs/reviews/atc-tutor-*-astra.md` (adversarial reviews).

## Task

In a Pipecat voice agent, explain a spoken air traffic control transmission (as transcribed by a
fine-tuned Nemotron ASR model) in one plain sentence of about 25 words, and answer short chat and term
questions as a tutor. Source data: 1,908 real ATCO2 transcripts owned by the team. Four days,
2026-10-09 → 10-12: 18 major full-SFT runs (excluding the canary and a partial run), 2 DPO runs, 10
export manifests (including the canary), roughly 14 dated rule declarations in the plan, and 13 Astra
review files.

## Arc

| Round | Main change | Outcome | Source |
|---|---|---|---|
| v1 | first corpus (shape-replicated synthetic, noise, chat) | review found leakage; relabelled exploratory | `docs/reviews/atc-tutor-data-training-astra.md` |
| v2 | rebuilt with full gates, contrast slice | **final test** (corpus gated against it by text and lineage; 381 of 387 references verified): tuned v2 with the short prompt 251/381 = 65.9% vs stock with the full prompt (its best on dev) 91/381 = 23.9%, +42.0 pts [34.9, 49.6]; unsupported 26.8% vs 64.0%; frontier model with the full prompt 58.8%. Same-family (Opus) judge; selected *configurations* compared; the dev diagnostic set had ancestry exposure | `docs/atc-tutor-results.md`, `docs/reviews/atc-tutor-final-astra.md` |
| v3, v4 | own-error repairs; targeted heading slice | did not beat v2 under the predeclared rule; differences within noise | `docs/atc-tutor-finetune-plan.md` |
| DPO pilot | 199 training + 23 validation own-error pairs | failed promotion on the short prompt; exploratory +3.4 (5 of 148 rows) on the full prompt | `docs/atc-tutor-results.md` |
| schedule study | 1 epoch decayed to zero × 3 seeds | −16 pts vs v2; seed SD 3.9. The short accelerated-decay schedule underperformed (not proof of underfitting in general) | `docs/atc-tutor-finetune-plan.md`, `docs/reviews/atc-tutor-rev6-astra.md` |
| v1.3 round 1 | contract v1.3 from owner audit; real train-side inputs | +14.3 pts on a newly withheld reserve vs v2 (contract, labels, data mix and selection all changed) | `docs/atc-tutor-finetune-plan.md` |
| round 2 | ~800 code-built core targets with a facts block; owner style | core 113/120, headings 38/38 | `artifacts/atc-tutor/bench/v13/core-v13ars2i486-short13.jsonl` |
| v1.3 DPO | on-policy pairs | −12 pts dev; answers vaguer | `docs/atc-tutor-finetune-plan.md` |
| round 3 | ASR-noise, emergency squawks, meta-talk | noisy inventions 32 → 14/80; mode errors 2 → 7; dev −2 | `docs/atc-tutor-finetune-plan.md` |
| round 4 | chat-vs-radio contrast pairs | mode errors 3; dev still below the gate | `docs/atc-tutor-finetune-plan.md` |
| round 5 | round-4 data at 831 updates (fixed exposure vs round 2: 649 × 4,295 / 3,356 = 830.6) | passed every gate: dev 67.3, severe 24.1, core 116/120, mode 1, noisy inventions 11/80, faithful 59/80. Single seed; supports but does not isolate the dilution hypothesis | `docs/atc-tutor-finetune-plan.md`, round 5 outcome |
| quant | FP8, NVFP4 vs same-stack BF16 (65.3) | FP8 64.0 passed (64.4 on H100); NVFP4 62.7 missed the −2.0 dev gate by 0.6 | `docs/atc-tutor-finetune-plan.md` |
| release | private HF repos (BF16 / FP8 / NVFP4), on the owner's explicit request | NVFP4 served locally on an RTX 5090 as the demo model, by the owner's choice | session history (owner request) |

## Owner contributions the teacher could not supply

- **Speaker convention:** in ATC phraseology a trailing callsign does not mark a pilot readback; the
  teacher had baked the opposite into 68 of 74 relevant targets and into references and judge. Ambiguous
  wording → "the speaker".
- **Callsigns as heard**; airline names from a checked telephony table; "probably" only for a clearly
  most-plausible garbled value (references then recorded licensed readings, so "probably" could not
  license invention).
- **Explain, never answer** a transmission addressed to someone else.
- Style: "confirms" rather than "reads back"; ~25 words; brief explanation of terms of art.
- **Fix in the model:** the webinar was about fine-tuning, so the owner ruled out runtime guards in the
  bot. That was this owner's constraint.
- Live rehearsals found what text suites missed: digit-by-digit headings, invented values on noisy ASR
  text, a misheard "wilco", meta-talk handling.

## Failure modes and responses

| Failure | Response that helped | What did not |
|---|---|---|
| Leakage (exact-string gates, chat exempt, descendants split independently) | gates on every turn, slot-masked + lineage, new label | — |
| Invented callsigns and registrations (top failure category) | "as heard" rule + checked table | restyle heuristics; teacher recall |
| Impossible values passed writer and critic (QNH 8,400 hPa) | deterministic range checks | cross-family critic alone |
| Transcription tags explained as words ("noise point Bravo") | contract rule, development reference re-freeze, regenerated targets | — |
| Long, messy real inputs | real train-side inputs (with other changes) | more clean synthetic shapes |
| Textbook coverage lost after a cleanliness quarantine | code-built core slice + facts block + deterministic checks | a heading slice without exactness checks |
| Noise data confused chat | minimal-pair contrast data | — |
| Larger corpus at fixed compute lost dev | fixed-exposure retrain | — |
| DPO vagueness | returned to SFT | — |
| Free-chat numeric facts (mode suite, judge `4079…`: v2 44 factual errors of 120 vs the frontier model's 10; under the later judge `fa7…`, round-2 candidates 24–39 vs frontier 12, `docs/atc-tutor-v13-r2-dev-report.md`) | partly: fact-checked term/number answers; demo steered toward phrases | — |

## In-house commands (shape only)

```bash
# Modal environment gb-rl, zorana-research-internal/llm-rl
modal volume put -e gb-rl gb-phonedata <export-dir> datasets/<label>
modal run -e gb-rl llm_rl/modal/nemo_sft.py::prep_phone_data --dataset-label <label> \
  --out-label <label>-s<seed>-nemo --shuffle-seed <seed> --max-total-tokens 2048
modal run --detach -e gb-rl llm_rl/modal/nemo_sft.py::train_full --run-label <run> \
  --data-label <label>-s<seed>-nemo --epochs <U*8/n> --seq-length 2048 --save-interval <N> \
  --skip-export --no-full-checkpoint-rotation
modal run -e gb-rl llm_rl/modal/nemo_sft.py::export_hf --run-label <run> --iteration <N> --out-name hf-iter<N>
# candidates side by side: phone_serving_app.py with GB_PHONE_LADDER=<comma-separated hf paths>
# quantization: atc_nvfp4_app.py (NeMo 26.08 pinned, ModelOpt pinned, MAMBA_MOE_*_AGGRESSIVE_CFG)
# release: atc_hf_publish_app.py (private repos, size verification, config drift allowlist)
```

Image `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` (Megatron-Bridge `nano` recipe); DPO used NeMo-RL
v0.6.0 (v0.7.0 crashed on Modal).

## Process facts

- Spend: about $590 overnight against an $800 cap; teacher spend about $500 by the next morning; judging
  was forecast at about $75 per round of ~5,100 answers (an estimate, not an itemized receipt).
- A pipeline chain without stop-on-failure stalled servers for 90 minutes once; a finished job was
  left idle for about 2 hours (as recorded in the session history).
- Reviews caught: leakage; a report declaring success on an incomplete cohort; a verifier that could not
  correct some fields; role errors counted correct; broken DPO pair builders; wrong recipe defaults; and
  "probably" as a licence to invent.
- The final-test claim was made once (v2); every later number is development-only.
