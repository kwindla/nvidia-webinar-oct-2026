# Training, Serving, Quantization, Release (Nemotron 3 Nano)

Model-specific settings for **Nemotron 3 Nano 30B-A3B** (hybrid Mamba-Transformer MoE, ~3.5B active
parameters) and the general practices around them. Values marked *starting point* worked in one project
([`case-study-atc.md`](case-study-atc.md)); verify flags against current NeMo, ModelOpt and vLLM releases.

## 0. Backend compatibility check (before any training)

Record, for the exact model and backend chosen:

- checkpoint repository and revision, architecture, license;
- the recipe that supports it (full SFT or PEFT) — via the user's stack or `nemotron-customize`'s catalog,
  artifact and hardware references and the selected live recipe; don't assume every backend supports
  every Nemotron variant;
- tokenizer and chat template actually used, reasoning mode (on/off) and tool-calling format;
- precision, maximum sequence length, packing support;
- measured memory and parallel topology for training and for serving;
- the artifact each stage produces and the next consumes (full HF weights, Megatron checkpoint, LoRA
  adapter, merged weights, quantized export) and the serving path for it.

**PEFT/LoRA:** declare rank, alpha, dropout and target modules; record the exact base revision and
adapter revision; declare whether you serve the adapter or a merged model. After a lossless merge or
export, check output equivalence on a fixed sample; after lossy quantization, check task quality against
declared tolerances instead (identical strings are not expected). Tensor-layout compatibility and quality
acceptance are separate checks. Preserve the SFT checkpoint's tokenizer and
template rather than reverting to the base model's.

The rest of this file gives Nemotron 3 Nano settings; for another size or variant, re-derive them from
its model card and the compatibility check.

## 1. Training (Nemotron 3 Nano, full SFT example)

**Initialization.** Start from NVIDIA's released chat-tuned checkpoint
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, converted to the trainer's format), not a raw base model,
unless the task needs otherwise. Start every comparable run from the same initialization.

**Method.** Full fine-tuning and LoRA/PEFT are both available in NVIDIA's stacks (Megatron-Bridge,
NeMo AutoModel; see `nemotron-customize`). Start from what the user's tested recipe supports (§0).

| Setting | Starting point (full SFT, one 8×B200 node) |
|---|---|
| Parallelism | TP2, PP2, sequence parallel, EP2, micro-batch 1, global batch 8 |
| Learning rate | 5e-6 peak, cosine to 0, 10 warmup iterations, decay over `train_iters` |
| Sequence length | from measured token lengths of rendered rows (short tasks fit in 2048), not a 32k recipe default |
| Updates | ~650–830 at GBS 8 for 3.3–4.3k per-turn examples (1.21–1.55 passes per example in the case study; recompute for your loader and packing) |
| Saves | every ~¼ epoch; weights only if storage is tight (then no true resume) |
| Replication | ≥2 seeds for decisions that matter |

Pipeline shape (any stack):

```text
upload export → prep (render chat template, mask, length filter; write receipt)
→ train (detached; log train_iters) → wait for full job exit
→ export the declared checkpoints to HF format → serve → evaluate
```

Gotchas:
- Step counts depend on the loader: per-turn expansion, packing, token-based batches and dropped
  partial batches all change them. In the case study's trainer `train_iters = ceil(epochs · n / GBS)`,
  so fractional epochs rounded up. Always check the logged value before waiting.
- Recipe defaults to override deliberately: very long sequence length, checkpoint rotation that saves
  optimizer state (doubles storage), a zero save interval falling back to the eval interval.
- Trainer validation loss on a few batches is noisy; never select on it.
- A schedule that decays to zero in one epoch can underperform a longer schedule at the same peak LR;
  if every run peaks at the same point, run a controlled schedule study (seeds × schedule) before
  concluding over- or underfitting.

## 2. Serving with vLLM

Nemotron 3 Nano has a reasoning mode. Decide in Stage 2 whether the task uses it, then train and serve
with the same render. For a no-reasoning task model, send it per request:

```python
client.chat.completions.create(
    model="<served-name>", messages=msgs, temperature=0, max_tokens=256, stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Server flags that mattered:

```bash
# --mamba-ssm-cache-dtype float32: precision of the hybrid Mamba state.
# --kv-cache-dtype bfloat16: set explicitly; a quantized export's hf_quant_config.json can otherwise
#   switch the KV cache to FP8 and change outputs between "BF16" and "quantized" arms.
vllm serve /model \
  --served-model-name <name> \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --mamba-ssm-cache-dtype float32 \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

- **The serving stack changes greedy outputs.** Identical BF16 weights scored ~2 points apart on dev
  across two vLLM versions in the case study, with nearly 30% of greedy answers different. Pin the serving
  version, KV dtype and GPU type for every comparison.
- Side-by-side candidates: one GPU each, a new deployment name per round, scaledown long enough to finish
  a benchmark, a warm-up request before measuring. Cold starts can take minutes.
- **GeForce GPUs in Docker** (e.g. RTX 5090): the vLLM image's CUDA forward-compat `libcuda` can fail
  with error 804; mask it with `--tmpfs /usr/local/cuda/compat`.
- **Verify which endpoint the product actually calls** (e.g. with the server's request counter). Some
  frameworks load `.env` files with override, so shell variables don't take effect.

## 3. Execution hygiene

- One chain per round with `set -euo pipefail`; each step asserts its own success marker; no step waits
  on an artifact a failed step never produced.
- Act on job completion immediately; idle finished jobs waste the schedule.
- Never `pkill -f` a pattern that appears in your own command line; select the PID by listening port.
- Watch shared-GPU memory: traffic to a co-located service while vLLM holds most of the GPU can crash it.
- **Cost accounting**, per item: measured training step time, allocation and startup, exports,
  concurrent serving hours, teacher/critic/reference/judge calls, storage. Historical figures are
  estimates; use current prices. In the case study, ~650 steps at ~2 s/step were ~22 minutes of step time on
  8×B200; startup, exports and side-by-side serving added substantially to each round's total.

## 4. Quantization (NVFP4 / FP8)

- **Prefer a supported native export path** for the model and format; **reuse a known recipe** if one
  exists; pin the container by digest and ModelOpt by version or commit.
- **Presets (Nano):** ModelOpt's Mamba-MoE configurations for NVFP4 or FP8, keeping NVIDIA's BF16
  exclusions (selected Mamba layers and projections) as in NVIDIA's published Nano builds. Other models
  need their own presets.
- **Calibration:** ~1,000 in-distribution rows (prompt plus target text), a fixed seed, and a receipt with
  hashes.
- **Layout check:** if NVIDIA publishes a quantized build of the same model and format, compare the
  export's tensor layout with it (exact header match) and normalize only where a verified compatibility
  repair is needed. If there is no published build, rely on the native export plus a serving load test.
  Never normalize a different model to Nano's layout.
- **Hardware:** NVFP4 needs Blackwell (B200, RTX PRO 6000, RTX 50-series); FP8 targets Hopper and Ada
  (H100, H200, L40S).
- **Acceptance vs same-stack BF16**, declared first; report pass/fail per gate. A near-miss is a fail; the
  owner may still choose to deploy it, recorded as an owner decision.
- Newer containers may need newer host drivers; a driver mismatch can surface as a CUDA initialization
  error.

## 5. Release (Hugging Face)

Only with explicit authorization (workflow Stage 15), at the recorded visibility (private if
unspecified).

- Upload from where the weights live, with the token read from `HF_TOKEN` in a secret store; never print
  tokens. If a user pastes a token into chat, say it is in the transcript and recommend rotating it.
- Create the repo at the recorded visibility, assert it, upload, verify every remote file by size, and
  load-check the uploaded checkpoint.
- Full-weight release: weights, index, config, tokenizer, chat template, custom modeling code, and the
  quantization config for quantized builds. Adapter release: adapter weights and config, the exact base
  model revision, and the tokenizer/template if changed.
- **Config drift:** compare the exported `config.json` with the base model's. Publish the base config only
  if every difference is on a reviewed allowlist of non-semantic keys; any other difference is a failure
  to investigate.
- Write your own model card (system prompt, serving command, a client example in the trained reasoning
  mode, hardware note, scores with caveats) and carry the base model's license and attribution; don't redistribute the
  vendor's card or assets.
- **Fine-grained read tokens are scoped per repository.** A reader sees a new repo only after the token's
  owner adds it. Check with the reader's credentials, and check anonymously that private means private.
