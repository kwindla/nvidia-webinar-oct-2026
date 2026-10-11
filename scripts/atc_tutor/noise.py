"""Live-like ASR noise slice: PocketTTS speech -> fine-tuned ASR (:8081) -> real errors.

collect: synthesize generated dev utterances with several voices, transcribe them
         with the live fine-tuned ASR, keep outputs that differ beyond case.
label:   Opus writes the explanation from the VISIBLE ASR text only (no access to
         the source utterance), then check.py form checks; rows go to noise.jsonl
         for export (critic pass via critic.py --input noise).

Usage:
  uv run python scripts/atc_tutor/noise.py collect --count 600
  uv run python scripts/atc_tutor/noise.py label
  uv run python scripts/atc_tutor/critic.py --input noise-labeled.jsonl --output noise-audited.jsonl
  uv run python scripts/atc_tutor/noise.py finalize
"""

import argparse
import asyncio
import io
import json
import random
import re

import aiohttp
import soundfile
import torchaudio

from check import check_row, normalize
from opus import OUT, ROOT, call_tool, client, ledger_summary

VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
ASR_URL = "http://127.0.0.1:8081/v1/audio/transcriptions"
CONTRACT = (ROOT / "docs/atc-tutor-target-contract.md").read_text()

LABEL_TOOL = {"name": "label", "input_schema": {
    "type": "object", "required": ["usable", "speaker", "explanation"],
    "properties": {
        "usable": {"type": "boolean", "description": "False if the text is too garbled to explain anything."},
        "speaker": {"enum": ["controller", "pilot", "unknown"]},
        "explanation": {"type": "string"},
    }}}
LABEL_SYSTEM = f"""You write the tutor's explanation for a speech-recognition transcript of
an ATC radio phrase, following this contract:

{CONTRACT}

You see exactly what the student model will see: the raw transcript, which may contain
recognition errors. Silently read sound-alike words, split or merged words and
misspelled phonetic letters as the intended standard terms. Never repair a number,
callsign or instruction from aviation priors: explain values as heard, and if
something material is unclear, explain the clear part and briefly say what is unclear.
Mark usable false only if nothing meaningful can be explained."""


def words(text):
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


async def collect(count, seed):
    from pocket_tts import TTSModel

    rows = [json.loads(l) for l in (OUT / "generated.jsonl").open()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    model = TTSModel.load_model()
    states = {voice: model.get_state_for_audio_prompt(voice) for voice in VOICES}
    path = OUT / "noise-candidates.jsonl"
    kept = tried = 0
    # The ASR server closes connections after each response; never reuse one.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
        for row in rows[:count]:
            voice = rng.choice(VOICES)
            audio = model.generate_audio(states[voice], row["utterance"])
            audio = torchaudio.functional.resample(audio.float().reshape(1, -1), model.sample_rate, 16000)
            buffer = io.BytesIO()
            soundfile.write(buffer, audio.squeeze(0).numpy(), 16000, format="WAV", subtype="PCM_16")
            form = aiohttp.FormData()
            form.add_field("file", buffer.getvalue(), filename="clip.wav", content_type="audio/wav")
            form.add_field("model", "atc02")
            async with session.post(ASR_URL, data=form) as response:
                text = (await response.json()).get("text", "")
            tried += 1
            if words(text) != words(row["utterance"]) and text.strip():
                kept += 1
                with path.open("a") as out:
                    out.write(json.dumps({"seed_id": row["seed_id"], "source": row["utterance"],
                                          "asr": text, "voice": voice}) + "\n")
    print(f"transcribed {tried}; kept {kept} with real recognition differences ({kept / max(tried, 1):.0%} yield)")


async def label(concurrency):
    candidates = [json.loads(l) for l in (OUT / "noise-candidates.jsonl").open()]
    api = client()
    gate = asyncio.Semaphore(concurrency)
    out_rows, stats = [], {"usable": 0, "unusable": 0, "form_reject": 0}

    async def run(candidate):
        # Training input mirrors the live pipeline after lowercasing.
        visible = candidate["asr"].lower().strip()
        async with gate:
            result = await call_tool(api, stage="noise_label", system=LABEL_SYSTEM, tool=LABEL_TOOL,
                                     user=json.dumps({"transcript": visible}))
        if not result["usable"]:
            stats["unusable"] += 1
            return
        check = check_row({"utterance": visible, "explanation": result["explanation"], "values_used": [], "bundle": {}})
        if check["status"] == "reject":
            stats["form_reject"] += 1
            return
        stats["usable"] += 1
        out_rows.append({"slice": "noise", "seed_id": candidate["seed_id"], "utterance": visible,
                         "explanation": result["explanation"].strip(), "speaker": result["speaker"],
                         "turns": [["user", visible, False], ["assistant", result["explanation"].strip(), True]]})

    await asyncio.gather(*(run(c) for c in candidates))
    (OUT / "noise-labeled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(stats)
    print("ledger:", json.dumps(ledger_summary()))


def finalize():
    """Audited noise rows that passed -> noise.jsonl (export input)."""
    rows = [json.loads(l) for l in (OUT / "noise-audited.jsonl").open()]
    kept = [r for r in rows if r["audit"]["verdict"] == "pass"]
    (OUT / "noise.jsonl").write_text("".join(json.dumps({
        "slice": "noise", "seed_id": r["seed_id"],
        "turns": [["user", r["utterance"], False], ["assistant", r["explanation"], True]],
    }) + "\n" for r in kept))
    print(f"noise rows kept {len(kept)} of {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "label", "finalize"])
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args.count, args.seed))
    elif args.command == "label":
        asyncio.run(label(args.concurrency))
    else:
        finalize()


if __name__ == "__main__":
    main()
