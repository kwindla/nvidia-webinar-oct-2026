# Nemotron ATC voice-agent demo

Local streaming ASR with a fine-tuned **Nemotron 3.5** checkpoint and Pipecat.
Includes a custom browser voice bot, the built-in dev console, and a synthetic-audio ASR smoke test.

## Components and architecture

| Component | Implementation |
|---|---|
| ASR server | NVIDIA [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp), Apache-2.0 C++ runtime, pinned **v0.2.0** CUDA release |
| Baseline ASR | Same NVIDIA runtime, current public **Nemotron 3.5 ASR Streaming 0.6B**, FP16 GGUF, CPU on :8082 |
| Judge | TypeSafe **Jev** (`jev-latest`, resolved `jev-1.13.0` in validation), independent asynchronous worker |
| ASR weights | Private `fciannella/atc02-nemotron-asr-finetuned`, converted locally from `.nemo` to FP16 GGUF |
| Pipeline | **Pipecat 1.12.0**, Python **3.12 only**; custom WebSocket adapter in `streaming_stt.py` |
| Bot speech | Deepgram **Flux TTS**, `flux-heather-en`, native Pipecat streaming service |
| Smoke-test speech | Kyutai **PocketTTS 3.3.0**, `alba` voice, local CLI |
| LLM | Switchable OpenAI **GPT-5.6 Terra** (`reasoning_effort="medium"`) / local **gb-phone**, streaming Chat Completions |
| Local LLM server | Existing user-managed **vLLM**, `http://127.0.0.1:8080/v1`, alias `phonellm-mini-a2rc1-nvfp4`; no credential required |
| Browser | React 19 / TypeScript 6 / Vite 8, Pipecat UI, Zustand 5, **SmallWebRTCTransport**, ngrok HTTPS |
| Client SDK | `client-js 1.13.1`, `client-react 1.8.2`, `small-webrtc-transport 1.10.8`; npm lockfile |
| Turn detection | **Silero VAD + Smart Turn v3**, interruption support and shared conversation context |

```text
Browser microphone → SmallWebRTC input → audio, in parallel to:
    Nemotron streaming STT (passes audio through without waiting for transcripts)
    Silero VAD + Smart Turn v3 (inside the user context aggregator; VAD stop also commits STT)
  → User context aggregator ends the turn once Smart Turn says done and the final transcript is in
  → LLM switcher (GPT-5.6 Terra / gb-phone / tuned ATC tutor) → Deepgram Flux TTS → SmallWebRTC output → Browser speaker
  → Assistant context aggregator

Same input PCM + commit boundaries → baseline ASR worker (:8082)
Fine-tune + baseline results → turn comparison → Jev worker → source-tagged RTVI → UI
```

ASR listens at `http://127.0.0.1:8081`; its streaming endpoint is
`ws://127.0.0.1:8081/v1/audio/transcriptions/realtime`. The adapter in
`streaming_stt.py` sends PCM and receives interim/final transcripts. VAD stop
commits ASR; Smart Turn decides when to respond after receiving the final text.
Server endpointing is disabled; right context is 3 (320 ms chunks).
The bot uses 16 kHz input and 24 kHz output. Deepgram Flux TTS streams LLM tokens
over its hosted WebSocket API; no local TTS server is required. PocketTTS remains
installed only for generating smoke-test audio. RTVI exposes transcripts and metrics.

## Install and prepare

Supported setup: Linux x86_64, NVIDIA GPU and CUDA-compatible driver, Git, FFmpeg,
and [uv](https://docs.astral.sh/uv/). Verified on an RTX 5090, driver 580.65.06.
Python is selected by `.python-version`; `uv.lock` pins dependencies. Run all
commands from this directory.

```bash
uv sync --locked
# Enter the private-repository token without writing it into a file/history:
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
uv run scripts/download_model.py
unset HF_TOKEN
uv run scripts/setup_nemo_cpp.py
```

The download pins HF revision `7b78fe63e6db3015ad8aa3df03422074f00dee73` and checks
the publisher's SHA-256. Setup verifies NVIDIA's release checksum, checks out
converter revision `6a3ca369370782acee0dd155e34394ba60b920c4`, and creates
`models/atc02/atc02-f16.gguf`. The 2.55 GB source checkpoint is retained.
No C++ build or Python NeMo installation is required. PocketTTS 3.3 is an
explicit dependency (the Pipecat `pocket-tts` extra currently pins 2.x). First-time setup needs
internet; PocketTTS also downloads public weights on its first run.

## Download the analysis dataset

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
uv run scripts/download_dataset.py
unset HF_TOKEN
```

Downloads private `fciannella/atc02-preview` into ignored
`datasets/atc02-preview/`, pinned to revision
`973e9352164fa66c445227b442b2fd71b2aad171`. Downloads resume on rerun;
the script checks all 1,911 files against their remote sizes (about 231 MB).

The dataset card identifies this as **ATCO2 Gold community evaluation**, not
the fine-tuning training set: 1,908 test utterances, 2.0004 hours, 16 kHz mono
WAVs with gold transcripts and provenance metadata. There is no training/dev
split; the publisher specifies evaluation-only use, not tuning or checkpoint
selection. The full split has known source-recording overlap with Jacktol.
See its local `README.md` for overlap and source-license details.
Analysis entry point: `datasets/atc02-preview/data/test/metadata.csv`, with
audio beside it. Verified all 1,908 metadata rows resolve to audio; all WAVs
are 16 kHz, mono, PCM16, totaling 2.00044 hours.

## Analyze audio and generate speaker noise

```bash
uv sync --locked --group analysis
uv run --group analysis scripts/analyze_audio.py
uv run --group analysis scripts/generate_noise.py
# Play locally on the machine running this command; start at low speaker volume.
ffplay -nodisp -autoexit artifacts/audio-analysis/noise-all.wav
# Another measured spectral profile, same reproducible random seed:
uv run --group analysis scripts/generate_noise.py --profile LSGS \
  --seconds 60 --rms-dbfs -30 --seed 42 --output artifacts/audio-analysis/noise-LSGS.wav
```

Analysis takes about a minute on the demo machine. It writes per-file measurements,
candidate-region timestamps, summary JSON, a spectral/level plot, and noise profiles to
ignored `artifacts/audio-analysis/`. All processing
is local. No source waveforms or transcripts are embedded in generated noise.

The generator creates 48 kHz mono PCM16 Gaussian noise with the measured spectral
color, controlled RMS and 50 ms fades; defaults: 60 seconds, −30 dBFS, seed 42.
Profiles: `all`, `LKPR`, `LKTB`, `LSGS`, `LSZB`, `LSZH`, `LZIB`, `YSSY`.
Its `.json` sidecar records generation settings. dBFS describes the WAV, not
speaker loudness or microphone SNR. Laptop speakers cannot reproduce the entire
low-frequency profile; speaker playback also cannot reproduce radio distortion
of the speech itself. Browser noise suppression/echo cancellation may reduce the
injected noise. Treat this as a prototype stress condition, not an ATCO2 simulator.

## Start ASR

In terminal 1:

```bash
./scripts/serve_asr.sh
```

In terminal 2, confirm readiness:

```bash
curl --fail http://127.0.0.1:8081/ready
```

Stop the server with Ctrl+C. The launcher uses the upstream binary unchanged;
`GET /health` reports its version. See NVIDIA's [server documentation](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/server.md)
for its HTTP and WebSocket APIs.

## Run the browser bot

Create `.env` from `.env.example` and set `OPENAI_API_KEY` and
`DEEPGRAM_API_KEY` with Flux TTS access (ignored by Git).
`DEEPGRAM_TTS_VOICE` defaults to `flux-heather-en`. The bot uses Pipecat’s
[`DeepgramFluxTTSService`](https://docs.pipecat.ai/api-reference/server/services/tts/deepgram)
with default token streaming and interruption handling. `flux_tts.py` guards
Pipecat 1.12 against late speech completions closing the next audio context after
a barge-in; the Flux connection and cross-turn voice state are preserved.
For this checkout, the keys were copied from `../pipecat/.env`; `.env` has owner-only
permissions. After moving the project, recreate its virtual environment with
`uv venv --clear --python 3.12 .venv` followed by `uv sync --locked`.
`NVIDIA_ASR_URL` defaults to `http://127.0.0.1:8081`. Start ASR as above, then:

```bash
uv run bot.py -t webrtc --host 127.0.0.1 --port 7860
```

Open `http://localhost:7860/client/`, choose **SmallWebRTC**, click **Connect**,
and allow microphone access. The bot greets you; speak naturally or try
“Repeat this clearance: United four fifty two, turn left heading two seven zero.”
Use headphones to avoid speaker echo. The console shows conversation, metrics,
and events. Disconnect ends the session; Ctrl+C stops the runner.

## Custom React UI

For local responses, start your existing vLLM deployment serving `gb-phone`
on port 8080. This project connects to that deployment; it does not install or
launch it. Verify it before selecting Local:

```bash
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gb-phone","messages":[{"role":"user","content":"Hi"}],"max_tokens":30}'
```

Optional `.env` settings: `LOCAL_LLM_URL=http://127.0.0.1:8080/v1` and
`LOCAL_LLM_MODEL=gb-phone` (the alias also works). The SDK uses a dummy key
for local requests; no OpenAI credential is sent to the local endpoint.

Requires Node.js 24+ and npm. With ASR and the Python bot running, start a third terminal:

```bash
npm --prefix client ci
# After the dataset analysis above, prepare synthetic browser audio assets:
uv run --group analysis scripts/prepare_web_noise.py
npm --prefix client run dev
```

Open `http://localhost:5173`. Click **Connect**, allow the microphone, and speak.
The UI includes live transcripts, bot audio visualization, microphone mute,
device selection, and typed messages. Microphone access is requested on connect;
if unavailable, the UI explains how to recover and text chat remains available.

The compact desktop layout keeps live transcript and ASR comparison visible
together, with each panel scrolling independently. The left **Voice agent** tab
contains connection, model, device and noise controls; **Model latency** contains
the rolling timing graphs. Tabs preserve the active session and noise playback.
The 32px Pipecat UI spectrum displays 40 frequency bands from the bot's actual
audio track, using a silent analysis output and resuming after browser gestures;
it rests during silence. On narrow screens the panels stack vertically.

Interim ASR text is lowercased to match the server's finalized output and shown
in italics. Each segment returns to regular type when finalized; already-final
segments and typed messages remain in regular type.
**Disconnect** ends the session; reconnect starts a fresh conversation.
Use headphones to avoid echo.

**Response model** lists the models the bot offers, from `GET /models` (Cloud always,
Local when :8080 answers, Tuned when `TUNED_LLM_URL` is set; names come from the bot,
and `TUNED_LLM_NAME`/`LOCAL_LLM_NAME` override them). Before connecting, a pick sets the
model the session starts on (sent as the `/start` body); otherwise sessions start on
`DEFAULT_LLM` (default `cloud`). If the picked model is no longer offered when the session
starts, the session starts on `DEFAULT_LLM` (or Cloud) instead, and the selector shows the
actual model once connected. After connecting, switching is confirmed by the server and
applies to the **next response** while any current reply finishes. All models share
conversation history, ASR, turn detection, and TTS. Switching to Cloud sends the
accumulated conversation—including turns handled locally—to OpenAI. Requests stay
server-side; model servers are not exposed through ngrok. Mid-session, an unavailable model
produces an error; there is no automatic fallback to the other provider.

`llm_switcher.py` wraps Pipecat's `LLMSwitcher`, choosing the service at each
context frame. It includes a Pipecat 1.12 workaround that deduplicates flush
probes per traversal leg, preventing typed-turn/interruption stalls in parallel
pipelines. Recheck this workaround when upgrading Pipecat. Regression checks:

```bash
uv run python -m unittest discover -s tests -v
```

The **background noise** panel plays generated noise directly in the browser,
including through ngrok. Choose All airports or an airport profile, press **Play
noise**, and adjust the volume (starts at 20%). **Stop noise** stops immediately;
changing profiles stops playback until Play is pressed again. Tracks loop every
60 seconds with brief fades. Playback is independent of the bot connection and
stops when leaving/reloading the page. It never autoplays.

Noise uses the browser's default output device, independently of Pipecat's speaker
selector. Choose laptop speakers in the OS output settings to feed noise into the
microphone. The slider is playback gain, not calibrated SNR. Browser suppression
can reduce the effect. The preparation command generates eight speech-free WAVs
in ignored `client/public/noise/`; Vite serves them in development and copies them
into `dist/` at build time. Rebuild after regenerating assets when using preview.
Only these synthetic tracks are served; the private dataset stays outside the web
root. Missing assets produce a playback error in the panel. Verified playback over
ngrok, volume changes, profile switching, stopping, reload without autoplay, and
missing-file error handling. Typechecking, ESLint, and the client tests pass.

The **model latency** panel uses Pipecat RTVI `onMetrics` events and Zustand.
Each model/metric keeps its last 30 reports in a compact sparkline, with a shared
zero-based millisecond scale. Cloud/local histories remain separate during model
switches; a new connection clears them. Empty services show no fabricated zeroes.

| UI metric | Reported measurement |
|---|---|
| Nemotron TTFS | Pipecat STT TTFB: speech end → finalized text, including VAD stop delay |
| LLM TTFB / TTFT | First service output / first answer token (`ttfat`, including any reasoning) |
| Deepgram Flux TTFB / TTFA | First audio bytes / first audible sample, including leading silence |

These are service reports, not complete-turn or browser playback latency.
Pipecat can emit TTFB when a request is interrupted; those reports remain visible.
Only emitted metrics appear; Smart Turn does not emit these latency types through
RTVI. Stable processor/model IDs distinguish each service. The STT adapter enables
Pipecat's native timing lifecycle and supplies its model metadata.

```bash
npm --prefix client run test
```

Parser tests cover units, model isolation, invalid samples, and the 30-report limit.
Previously verified real ASR, both LLMs, and PocketTTS reports through a browser voice session,
plus desktop/mobile layouts. Screenshots: ignored `artifacts/latency-*.png`.

The UI adapts [pipecat-deepgram-webinar-demo](https://github.com/pipecat-ai/pipecat-deepgram-webinar-demo/tree/e523272fc8474a67dee44f378acdbfa086b4c80d/client).
Pipecat UI registry components live in `client/src/components/pipecat/`;
`components.json` retains the official [Pipecat UI](https://ui.pipecat.ai) registry.
The original Deepgram STT, reservation tools, Daily transport, and deployment configuration
were removed. Base UI styles retain their MIT notice in `client/SHADCN-LICENSE`;
the bundled font retains its OFL license.

`src/stores/session.ts` creates one Pipecat client per page and uses Zustand for
session actions, errors, and speaking status. Pipecat's React provider owns
conversation, transport, and device state. React effects are reserved for
media/DOM synchronization in the UI components; transcript scrolling uses a
callback ref with cleanup.

```text
Browser :5173 → Vite proxy → Pipecat runner :7860
  POST /start → session + ICE configuration
  /sessions/<id>/api/offer → SmallWebRTC signaling
Browser ← direct WebRTC audio + RTVI events → Python pipeline
```

Development and preview use the same proxy and transport. No frontend environment
file or API key is needed; the OpenAI key stays in the Python server's `.env`.
Cloud responses send conversation context to OpenAI. For a production-build smoke test:

```bash
npm --prefix client run lint
npm --prefix client run build
# Stop the dev server first; preview also uses port 5173.
npm --prefix client run preview
```

For HTTPS microphone access from another machine:

```bash
ngrok http http://127.0.0.1:5173
```

Open the printed HTTPS URL at its root (no `/client/` suffix). The built-in console
can be tunneled the same way (`ngrok http http://127.0.0.1:7860`, then open
`/client/`). Both use the same bot. Ngrok carries UI/signaling, while WebRTC media uses
ICE directly, with Cloudflare TURN as a relay fallback when configured (see below).
Vite allows `*.ngrok.app`; add an explicit hostname in `client/vite.config.ts`
if using a different tunnel domain. Preview is for this local demo, not a
production hosting configuration.

A second UI can front a second bot (such as the fully local copy on :7871) by setting
the proxy target and port:

```bash
BOT_URL=http://127.0.0.1:7871 CLIENT_PORT=5174 npm --prefix client run preview
ngrok http http://127.0.0.1:5174
```

Validated through the custom HTTPS UI: synthetic microphone audio → Nemotron
transcript → Smart Turn → GPT-6 Luna → PocketTTS audio (before the Flux switch) received over WebRTC;
typed messages, disconnect/reconnect, microphone denial, and responsive layout.
Also verified Cloud → Local → Cloud recall of earlier turns, switching while
the bot speaks, and synthetic microphone → local LLM → PocketTTS playback.
Build/typechecking and ESLint pass. Browser screenshots are in ignored
`artifacts/custom-ui-*.png`.

## Run the synthetic-audio demo

With ASR running, in terminal 2:

```bash
mkdir -p artifacts
uv run pocket-tts generate \
  --text 'United four fifty two, turn left heading two seven zero. Descend and maintain five thousand feet.' \
  --voice alba --output-path artifacts/synthetic.wav
ffmpeg -hide_banner -loglevel error -y -i artifacts/synthetic.wav \
  -ar 16000 -ac 1 -c:a pcm_s16le artifacts/synthetic-16k.wav
uv run smoke_test.py
```

The test requires streaming partial events and a finalized transcript matching
the expected words, ignoring case and punctuation. Errors, timeouts, or mismatches
fail the command. Results: `artifacts/smoke-result.json`. To use another sample:

```bash
uv run smoke_test.py --audio path/to/mono-16k.wav --expected 'Expected spoken words'
```

Verified: the 6.28-second PocketTTS sample produced the correct final transcript
and streaming partial events. This is an integration check, not an accuracy or
latency benchmark. Deterministic speech boundaries do not test automatic VAD.

## ATC tutor fine-tune

The fine-tuned Nemotron 3 Nano 30B-A3B explains a spoken ATC transmission in one plain
sentence.

**Final test** (387 held-out real transcripts, locked one-shot run):

| Arm | Fully correct | Unsupported claims |
|---|---|---|
| **Tuned model** | **65.9%** | 26.8% |
| Stock Nano | 23.9% | 64.0% |
| GPT-5.6 Terra | 58.8% | 19.9% |

Target contract: [docs/atc-tutor-target-contract.md](docs/atc-tutor-target-contract.md).
The evaluator is frozen at [docs/atc-tutor-eval-contract.md](docs/atc-tutor-eval-contract.md).
Training uses the PhoneLLM Modal recipe in `../zorana-research-internal`.

### Current demo state (2026-10-11 evening)

**Running locally:**
- fine-tuned ASR on :8081 and the round-5 NVFP4 tutor on :8083 (both on the GPU);
- base ASR on :8082 (CPU);
- bot on :7871 (7861 is used by another project), behind the React UI on :5174
  (`vite preview` with `BOT_URL`), tunneled by ngrok.

**Stopped:** the earlier :7860 bot and :5173 UI, and the local Mini vLLM `gb-phone`
(:8080), which does not fit beside the tutor; the bot offers **Local** only when it answers.

**Tuned model:** the round-5 NVFP4 build served locally (see below; `.env`
`TUNED_LLM_URL=http://127.0.0.1:8083/v1`, `TUNED_PROMPT=v13`). Earlier candidates remain
on Modal, scaled to zero: round 2 `atc-tutor-v13ar-s2-sft/hf-iter486` at
`https://daily-gb-rl--atc-tutor-serving-v13-phone.modal.run/v1` (`TUNED_PROMPT=v13`), and
the final-tested v2-328 at `https://daily-gb-rl--atc-tutor-serving-phone.modal.run/v1`
(`TUNED_PROMPT=v2`). DPO checkpoints are not used in the demo.

**Before a session:** restart the bot within 24 hours (Cloudflare TURN credentials
expire). Pick **Tuned** before connecting, or switch after the Cloud greeting; the tutor
greets oddly with an empty conversation, so starting on Cloud reads better.

### Fully local tuned model (RTX 5090 / Blackwell)

The round-5 NVFP4 build fits next to the fine-tuned ASR on one 32 GB GPU (19 GB weights,
`GPU_MEM_UTIL=0.80` by default). Download it (needs a token with read access to the
private repo) and serve it with vLLM 0.31.0 in Docker on :8083:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo; export HF_TOKEN
uvx --from huggingface_hub hf download daily-co/atc-tutor-nano-r5-nvfp4 \
  --local-dir ~/local-models/atc-tutor-r5-nvfp4
unset HF_TOKEN
./scripts/serve_asr.sh            # terminal 1, unless already running
./scripts/serve_tutor_nvfp4.sh    # terminal 2; ready when curl -f http://127.0.0.1:8083/health succeeds
```

Point the bot's **Tuned** arm at it in `.env` (Pipecat's runner reloads `.env` with
override, so shell variables do not work): `TUNED_LLM_URL=http://127.0.0.1:8083/v1`,
`TUNED_PROMPT=v13`; `TUNED_LLM_API_KEY` may be any value. Then start the bot:

```bash
uv run bot.py -t webrtc --host 127.0.0.1 --port 7871
```

Open `http://localhost:7871/client/`, or the custom UI through ngrok (see Custom React UI:
`BOT_URL`/`CLIENT_PORT`). ASR, the tutor LLM, VAD and turn detection run
locally; Deepgram TTS, the Jev judge, the **Cloud** arm and TURN still use the network.

### Use the tuned model in the voice demo

The selected checkpoint (`atc-tutor-v2-sft/hf-iter328`) is served by the Modal app
`atc-tutor-serving`. To use it, set in `.env`:
- `TUNED_LLM_URL=https://daily-gb-rl--atc-tutor-serving-phone.modal.run/v1`
- `TUNED_LLM_API_KEY`: the `gb-serving-key` value.
- `TUNED_PROMPT`: `v2` for v2/v3 models (default), `v13` for v1.3 models.

The UI then shows a **Tuned** model button. A cold start takes about 400 s, so warm the
endpoint before a session:

```bash
uv run python scripts/atc_tutor/probe_endpoint.py https://daily-gb-rl--atc-tutor-serving-phone.modal.run ../zorana-research-internal/.env
uv run python scripts/atc_tutor/live_smoke.py   # 20 turns: PocketTTS -> fine-tuned ASR (:8081) -> tuned endpoint
```

### Data pipeline

Prerequisites:
- `ANTHROPIC_API_KEY` in `../pipecat/.env`, for Claude Opus 5.5 at high effort;
- the downloaded dataset;
- the fine-tuned ASR on :8081, needed only for `noise.py collect` and `live_smoke.py`.

Outputs go to the ignored `artifacts/atc-tutor/`, and every Opus call is logged with
its cost in `ledger.jsonl`.

```bash
cd scripts/atc_tutor
uv run python split.py                 # dev / selection / final-test split by recording
uv run python mine.py reserve          # frozen dev partition: devdiag (diagnostic) / devpool
uv run python tag.py                   # clause taxonomy for dev rows
uv run python generate.py              # shape-quota variants with fresh values
uv run python check.py --self-test && uv run python check.py
uv run python critic.py                # critic + independent adjudicator
uv run python generate.py --contrast 200   # abbreviated callsigns, low flight levels
uv run python check.py --input generated-contrast.jsonl --output checked-contrast.jsonl
uv run python critic.py --input checked-contrast.jsonl --output audited-contrast.jsonl
uv run python targeted.py --count 250  # heading / heading-vs-runway slice (then check + critic as above)
uv run python chat.py && uv run python chat.py --audit
uv run python noise.py collect && uv run python noise.py label
uv run python critic.py --input noise-labeled.jsonl --output noise-audited.jsonl
uv run python mine.py repair --arm <policy-arm>   # own-error repairs (after bench runs on devpool)
uv run python restyle.py               # registration restyle self-tests (contract v1.2)
uv run python export.py --label atc-tutor-v3 --restyle --exclude-devdiag --repairs repairs-<arm>.jsonl
```

### Train, export and serve

Run these from the PhoneLLM repo root (`v2` produced the selected model):

```bash
M=llm-rl/.venv/bin/modal
$M volume put -e gb-rl gb-phonedata <export dir> datasets/<label>
$M run -e gb-rl llm-rl/llm_rl/modal/nemo_sft.py::prep_phone_data \
  --dataset-label <label> --out-label <label>-nemo --shuffle-seed 1234 --max-total-tokens 2048
$M run --detach -e gb-rl llm-rl/llm_rl/modal/nemo_sft.py::train_full \
  --run-label <run> --data-label <label>-nemo --epochs 2 \
  --seq-length 2048 --save-interval <steps> --skip-export --no-full-checkpoint-rotation
# After the run finishes (single-node runs commit the volume at exit):
$M run -e gb-rl llm-rl/llm_rl/modal/nemo_sft.py::export_hf --run-label <run> --iteration N --out-name hf-iterN
# Up to eight checkpoints side by side (ladder0-ladder7), or one for the demo:
GB_PHONE_APP_NAME=atc-tutor-bench-serving GB_PHONE_FAMILY=nano \
GB_PHONE_LADDER="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16,/data/checkpoints/<run>/hf-iterN,..." \
  $M deploy -e gb-rl llm-rl/llm_rl/modal/phone_serving_app.py
GB_PHONE_APP_NAME=atc-tutor-serving GB_PHONE_FAMILY=nano GB_PHONE_MODEL_PATH=/data/checkpoints/<run>/hf-iterN \
  $M deploy -e gb-rl llm-rl/llm_rl/modal/phone_serving_app.py
```

### Benchmark

Rubric v2 (the frozen v1.1 evaluator). References are written and resolved before any
candidate is scored, and the final-test split runs once, after the model and prompt
are locked.

```bash
cd scripts/atc_tutor
uv run python bench.py references --split devdiag && uv run python bench.py resolve --split devdiag
uv run python bench.py run --split devdiag --arm <name>:short --url <ladder url> --checkpoint <path>
uv run python bench.py run --split devdiag --arm terra:full --kind openai
uv run python bench.py judge --split devdiag
uv run python bench.py report --split devdiag --baseline stock:full --candidate <name>:short
uv run python jev_check.py --split selection --arms <arm> ...   # diagnostic cross-judge
```

Failure analysis of a judged arm on the development splits (never the final test):
judge test-retest, an Opus failure taxonomy and a human-audit sample.

```bash
uv run python analyze.py rejudge --arm v2i328-short
uv run python analyze.py classify --arm v2i328-short
uv run python analyze.py report --arm v2i328-short   # writes artifacts/atc-tutor/analysis/<arm>/report.md
uv run python analyze.py audit --arm v2i328-short --size 100
```

### v1.3 pipeline (contract `docs/atc-tutor-contract-v13.md`)

v1.3 trains on real training-side transcripts as well as synthetic ones, with
callsigns as heard, a checked telephony table, "probably" readings and
explain-versus-answer pairs. Opus writes and GPT-5.6 Terra critiques (`OPENAI_API_KEY`
in `.env`).

```bash
cd scripts/atc_tutor
uv run python split13.py                     # immutable split: train / reserve / dev cohorts
uv run python telephony.py                   # telephony-v13.json from OpenFlights + curated entries
uv run python data13.py targets --source real --writer opus
uv run python data13.py targets --source synthetic --writer opus
uv run python data13.py variants && uv run python data13.py targets --source variant --writer opus
uv run python modes13.py                     # mode pairs, term answers, chat
uv run python export13.py --label atc-tutor-v13a --arm A   # --arm B adds the variants
uv run python bench13.py references --cohort dev           # then judge / report, see the eval doc
# round 2: core phraseology and the owner style pass
uv run python core13.py && uv run python data13.py targets --source core --writer opus
uv run python restyle13.py                   # writes targets-*-r.jsonl
uv run python export13.py --label atc-tutor-v13ar --arm A --restyled
# round 3: recognizer noise, emergency squawks, acknowledgements, meta-talk
uv run python noisy13.py check               # frozen 80-item noisy check (once)
uv run python noisy13.py corrupt && uv run python noisy13.py special
uv run python noisy13.py targets --source noisy && uv run python noisy13.py targets --source special
uv run python noisy13.py meta
uv run python export13.py --label atc-tutor-v13r3 --arm A --round3
uv run python noisy13.py score --arm NAME:short13 --url https://<serving app>.modal.run
# round 4: chat-vs-radio contrast pairs (round 5 retrains this corpus to 831 updates)
uv run python noisy13.py contrast
uv run python export13.py --label atc-tutor-v13r4 --arm A --round4
```

Training uses the same Modal commands as above with the v1.3 labels; the run
parameters are in the plan's v1.3 declaration. Round 5 is `train_full --data-label
atc-tutor-v13r4-s2-nemo --epochs 1.5471 --save-interval 311` (831 updates).

### Quantized builds (NVFP4 and FP8)

NVFP4 needs Blackwell GPUs (B200, RTX PRO 6000, RTX 50-series); FP8 runs on Hopper and
Ada (H100, H200, L40S). Both use the PhoneLLM Mini recipe: NVIDIA's Mamba-MoE ModelOpt
preset and BF16 layer policy, 1,000 calibration prefills from the training rows, and
normalization to NVIDIA's published Nano checkpoint layout. From the PhoneLLM checkout
(branch `atc-tutor-dpo`):

```bash
llm-rl/.venv/bin/modal run --detach -e gb-rl llm-rl/llm_rl/modal/atc_nvfp4_app.py \
  --run atc-tutor-v13r5-nvfp4-a --source atc-tutor-v13r5-s2-sft/hf-iter831 \
  --calibration atc-tutor-v13r4-s2-nemo/training.jsonl          # add --fmt fp8 for FP8
# serve on vLLM 0.31 (BF16 KV cache); GB_PHONE_GPU=H100 for FP8 on Hopper
GB_PHONE_APP_NAME=atc-tutor-bench13q-serving GB_PHONE_FAMILY=nano \
  GB_PHONE_IMAGE=vllm/vllm-openai:v0.31.0 GB_PHONE_GPU=RTX-PRO-6000 \
  GB_PHONE_EXTRA_VLLM_ARGS='["--kv-cache-dtype","bfloat16"]' \
  GB_PHONE_LADDER=/data/checkpoints/atc-tutor-v13r5-s2-sft/nvfp4-atc-tutor-v13r5-nvfp4-a \
  llm-rl/.venv/bin/modal deploy -e gb-rl llm-rl/llm_rl/modal/phone_serving_app.py
```

Results against BF16 on the same stack are in the plan's quantization section: FP8 passes
on RTX PRO 6000 and H100; NVFP4 is 2.6 points lower on dev.

### Hugging Face release (private)

Round 5 is published from Modal (secret `hf-write-token`) to private repos
`daily-co/atc-tutor-nano-r5-{bf16,fp8,nvfp4}`, each with a model card giving the system
prompt and vLLM usage. Partner read tokens are fine-grained: add each repo to the token
(Hub → Settings → Access Tokens → Repositories permissions → read) before sharing.

```bash
# from the PhoneLLM checkout's llm-rl/ (branch atc-tutor-dpo); --dry-run prints the card
uv run modal run -e gb-rl -m llm_rl.modal.atc_hf_publish_app --fmt bf16   # or fp8, nvfp4
```

### Optional DPO stage (exploratory)

`dpo_build.py` turns a policy's own devpool errors and their verified repairs into
NeMo-RL DPO pairs, rendered by the real prep template. The config is
`llm_rl/grpo/configs/dpo_atc_v1.yaml` on the PhoneLLM branch `atc-tutor-dpo`. The stage
was not used for the final model.

```bash
uv run python scripts/atc_tutor/dpo_build.py stage --policy-arm v2i328-short --corpus atc-tutor-v2 --label atc-dpo-v1
# upload artifacts/atc-tutor/dpo/atc-dpo-v1/*.chat.jsonl to gb-phonedata datasets/atc-dpo-v1-stage, then
#   prep_phone_data --dataset-label atc-dpo-v1-stage --out-label atc-dpo-v1-rendered
# and download training.jsonl, validation.jsonl, rows_meta.jsonl into that directory
uv run python scripts/atc_tutor/dpo_build.py pairs --label atc-dpo-v1
# upload *.dpo.jsonl as gb-rl-sft atc-dpo-v1/{train,validation}.jsonl, then from llm-rl/:
GB_NEMO_RL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.6.0 GB_NEMO_RL_RUNS_VOLUME=gb-nemo-rl-runs-atc-dpo1 \
  .venv/bin/modal run --detach -e gb-rl llm_rl/modal/nemo_rl_app.py::train_dpo \
  --run-label atc-dpo-v1 --config /root/llm_rl/grpo/configs/dpo_atc_v1.yaml
# export a step (with the same GB_NEMO_RL_RUNS_VOLUME as training):
#   nemo_rl_app.py::export_a6_checkpoint --run-label atc-dpo-v1 --step N --out-name atc-dpo-v1-sN
```

## Skills

`skills/nemotron-nano-task-finetune/` is a coding-agent skill that generalizes the
method used for the ATC tutor fine-tune (contract and baseline, frozen splits and
predeclared rules, audited teacher data, SFT, multi-suite judging, quantization,
authorized release). Start with its `SKILL.md`; the ATC run is its worked example
in `references/case-study-atc.md`. To use it in Claude Code, copy or symlink the
directory into `.claude/skills/` (project) or `~/.claude/skills/` (user).

## Project maintenance

Keep this README current with every change, including dependency, server, and
demo commands; remove obsolete code/configuration. `AGENTS.md` records this rule.
Use the Pipecat context hub for current APIs and best practices. Credentials,
`models/`, `datasets/`, `artifacts/`, `.tools/`, and `vendor/` are excluded by `.gitignore`.

## Dual-ASR comparison

The fine-tuned EA model remains the only ASR input
to Smart Turn, the LLM, and TTS. Two session-owned asyncio workers run the baseline
WebSocket and Jev independently; no baseline transcription frames enter Pipecat's
conversation pipeline. Custom RTVI snapshots carry source/model IDs, a session ID,
revision, turn IDs, and separate interim/final text to the comparison panel.

Prepare the public baseline (about 2.37 GB source + 1.33 GB converted model):

```bash
uv sync --locked
uv run scripts/setup_base_asr.py
# Separate terminal; keep the fine-tuned server on 8081 running too:
./scripts/serve_base_asr.sh
curl --fail http://127.0.0.1:8082/ready
```

Pins HF revision `ea30d66debe3740a08b573244286791d423d6b3e`, verifies the source
SHA-256, and uses the same pinned FP16 converter as the fine-tune. The baseline
runs on CPU because the GPU also hosts the user-managed local LLM. The launcher pins it to
logical CPUs 0–7; set `BASE_ASR_CPUS` to change the affinity (requires `taskset`). Both servers
use 320 ms right context, disabled endpointing, and identical English/verbatim
session settings. CPU versus GPU latency and public-versus-EA checkpoint versions
are confounds: this demo does not isolate the effect of fine-tuning alone.

Add to the project's ignored `.env` (all options are in `.env.example`):

```dotenv
BASE_ASR_URL=http://127.0.0.1:8082
TYPESAFE_API_KEY=your-typesafe-key
JEV_MODEL=jev-latest
```

Restart the bot with the usual command above. In **ASR comparison**, open
**Reference and judging settings**, enter what you intend to say, and **Apply for
next turn**. The reference is armed once and consumed by the next microphone
turn. Pauses that Smart Turn considers complete split the phrase into separate
turns; only the first gets the reference. Both columns preserve finalized text
and show lowercase italic interims. The last 20 turns are retained per connection.

Jev judging is **on by default**. Each turn gets a badge: **Fine-tuned better**,
**Base better**, **Tie**, **Insufficient evidence**, or **Inconclusive** when the
candidate-order checks disagree. Pending and unavailable judgments are explicit.
Identical normalized transcripts receive a local tie without an API call.

The worker sends anonymous transcript pairs, an optional reference, and the last
**10 user/assistant speaker turns** from the LLM context to the official TypeSafe
endpoint. Context is copied at speech start, excludes system/tool/media content,
and caps each prior turn at 2,000 characters. This prevents the current transcript
or its assistant response from leaking into its own judgment. Prior ASR text may
still be wrong; the rubric warns against treating repetition as proof. Context
judging can be disabled in the settings.

Jev returns a **context-based preference**, not verified acoustic correctness.
A supplied reference takes priority and adds locally computed raw WER. Two isolated
calls reverse the anonymous candidates; only agreement yields a preference.
The rubric is in `asr_comparison.py:preference_question`, following the official
[TypeSafe skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md)
and [Choice documentation](https://docs.typesafe.ai/primitives/choice).

Baseline failures, commit-ack delay >8 seconds after the baseline sends a commit,
queue overflow, and missing Jev credentials produce explicit unavailable states
without stopping the voice bot. A primary-ASR connection failure is fatal to the
voice session. Reference length is capped at 500 characters; transcript text is
capped at 2,000 characters per source/turn, with over-limit comparisons rejected.
Jev has a 10-second wall-clock timeout, no SDK retries, four queued jobs per session,
two concurrent requests per bot process, and no lifetime turn limit. Local WER survives missing/failed Jev judgments; it is unavailable when an ASR
result is missing or the pair times out.
A lack of interim text is not treated as lag: noise can be VAD-positive without
producing words. Over-60-second turns are unjudgeable. Disconnect cancels workers;
WebSocket close handshakes are limited to one second. Reconnect starts a fresh comparison. Standalone typed
messages are not microphone comparison turns.

Validation:

```bash
uv run python -m unittest discover -s tests -v
npm --prefix client run test
# Requires both ASR servers, .env Jev key, and synthetic-16k.wav from the demo above:
uv run scripts/smoke_comparison.py
# Jev-only known-answer checks (no ASR servers needed):
uv run scripts/smoke_jev.py
```

The live smoke writes `artifacts/comparison-smoke.json`. It verifies two finalized
transcripts and a completed judge result, allowing the models to disagree. The
baseline's existing exact-text smoke fails because it adds “hundred” to “four
fifty two”; that is an observed output difference, not a streaming failure.

Measured unpaced CPU base RTF: 0.134 (0.844 seconds for 6.28 seconds of audio), with
the bot running. The first browser comparison test retained the fine-tune-only
LLM context and, before the Flux switch, received PocketTTS TTFB of 74/123 ms, versus earlier 81/95 ms.
These small samples establish basic viability, not a performance benchmark or a
zero-contention guarantee. Both actual servers and live Jev were exercised.

Final checks: 18 Python tests, 4 client tests, typecheck/build and lint pass.
Claude re-reviewed the fixes with no remaining blocker/high findings. The live
Jev checks pass for number errors, omissions, numeric formatting, and agreement.
A forced baseline-server failure left the bot connected and answering typed
messages; the baseline was restored afterward.

Current Flux/context-judge validation: 23 Python tests and 5 client tests pass,
as do the frontend build/lint and seven live Jev known-answer checks. Browser
checks cover Flux audio/spectrum, latency reports, ASR turn badges and interruption.
The public providers briefly returned Jev 503 and Flux capacity errors during
validation; later requests recovered. Provider failures remain visible in the UI.

### Cloudflare TURN for off-LAN connections

SmallWebRTCTransport remains the media transport; ngrok carries the webpage and
HTTP signaling. Cloudflare provides a TURN relay fallback for restrictive NATs.
Set `CLOUDFLARE_TURN_KEY_ID` and `CLOUDFLARE_TURN_API_TOKEN` in `.env` (ignored,
owner-only). Start the bot with the command above, without `--ice-servers`.
Startup mints temporary credentials and configures Pipecat's standard runner via
`PIPECAT_ICE_SERVERS`; `/start` supplies the same ICE configuration to the browser.
The permanent API token never reaches the browser. UDP and TCP TURN plus TLS/443
are enabled. Without Cloudflare configuration the demo uses public STUN only.

**Restart the bot daily:** temporary TURN credentials expire 24 hours after
startup; this demo does not renew an active connection's credentials. Reconnect
clients after restarting. Configured credential-generation failures stop startup
rather than silently falling back to LAN-only connectivity. Keep `.env`, logs,
and generated artifacts out of Git; `.env.example` contains empty placeholders.

Verified with browser relay-only connections over both UDP and TLS/443; both
received bot audio. Refresh the UI before testing from a different network.

The voice agent system instruction is defined in `bot.py` and shared by both LLMs.
The greeting is used only before the first actual user message.
The user speaks an aviation phrase; the agent gives only one short explanation
sentence, without repeating or quoting the phrase. It opens with “Let’s practice aviation radio calls. Say an
aviation phrase, and I’ll briefly explain what it means.”
The cloud model allows
4,096 completion tokens, including reasoning, so medium reasoning has room to
produce a spoken answer.

The system instruction includes ten fictional few-shot examples: seven controller
instructions covering flight levels, altitudes, headings, transponder codes, radio
frequencies, holding short, and landing with wind information, plus three pilot
readbacks covering climb, holding short, and landing clearance. Each response
consists of one short explanatory sentence that
identifies the speaker's role. The bot remains a tutor; it does not respond as the
other participant in the radio exchange. These are instructional examples, not
conversation history or dataset quotations. Terminology was checked against the
[FAA Pilot/Controller Glossary](https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/)
and [radio communication guidance](https://www.faa.gov/air_traffic/publications/aim_html/chap4_section_2.html).

The voice-agent panel has a labeled **Mute microphone / Unmute microphone**
button below Connect. It controls the Pipecat client microphone and is disabled
until connected with microphone access; bot audio and text chat remain available
while muted. The adjacent device menu selects the microphone.
