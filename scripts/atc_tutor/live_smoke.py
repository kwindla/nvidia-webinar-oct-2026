"""Multi-turn smoke of the tuned tutor on the live request path (no browser).

Each user turn is synthesized with PocketTTS, transcribed by the fine-tuned ASR
(:8081 batch endpoint) and lowercased as the bot does, then sent to the tuned
endpoint with the bot's exact settings (short tutor prompt, no-think, 250-token cap)
and the accumulated conversation history. Reports per-turn time to first token and
the replies. Browser/WebRTC/TURN audio and barge-in still need a human rehearsal.

Usage: uv run python scripts/atc_tutor/live_smoke.py [--url ...] [--prompt v13] [--out live-smoke-v13.json]
"""

import argparse
import asyncio
import io
import json
import statistics
import time

import aiohttp
import soundfile
import torchaudio
from dotenv import dotenv_values
from openai import AsyncOpenAI

from opus import OUT, ROOT
from export import SHORT_PROMPT as TUTOR_PROMPT  # shared tutor_prompts.py, hash-checked
from tutor_prompts import SHORT13_PROMPT

TURNS = [
    "hello",
    "qantas four ninety eight climb flight level one two zero",
    "what does flight level mean",
    "speedbird two one turn left heading two seven zero",
    "say that again please",
    "hotel bravo kilo oscar xray runway two five cleared to land wind two four zero degrees one two knots",
    "climbing flight level one eight zero swiss three one yankee",
    "jetstar seven sixty seven hold short runway three four left",
    "what is a holding point",
    "oscar kilo papa romeo mike after bukap descend to four thousand feet",
    "lufthansa four two contact tower one one eight decimal one",
    "roger",
    "cathay six squawk four two seven one",
    "wilco",
    "victor hotel bravo victor papa expect ils approach runway one six right",
    "thanks that was helpful",
    "easy two one eight line up runway two eight and wait",
    "air france one zero eight zulu negative not cleared to land go around",
    "what can you do",
    "goodbye",
]


async def transcribe(session, model, state, text):
    audio = model.generate_audio(state, text)
    audio = torchaudio.functional.resample(audio.float().reshape(1, -1), model.sample_rate, 16000)
    buffer = io.BytesIO()
    soundfile.write(buffer, audio.squeeze(0).numpy(), 16000, format="WAV", subtype="PCM_16")
    form = aiohttp.FormData()
    form.add_field("file", buffer.getvalue(), filename="turn.wav", content_type="audio/wav")
    form.add_field("model", "atc02")
    async with session.post("http://127.0.0.1:8081/v1/audio/transcriptions", data=form) as response:
        return (await response.json()).get("text", "").lower().strip()


async def main():
    from pocket_tts import TTSModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=dotenv_values(ROOT / ".env").get("TUNED_LLM_URL"))
    parser.add_argument("--prompt", choices=["v2", "v13"], default="v2", help="the prompt the model was trained with")
    parser.add_argument("--out", default="live-smoke.json")
    args = parser.parse_args()
    api = AsyncOpenAI(base_url=args.url, api_key=dotenv_values(ROOT / ".env")["TUNED_LLM_API_KEY"], timeout=900)
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt("alba")
    messages = [{"role": "system", "content": SHORT13_PROMPT if args.prompt == "v13" else TUTOR_PROMPT}]
    log, ttfts = [], []
    # The ASR server closes connections after each response; never reuse one.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
        for index, spoken in enumerate(TURNS):
            heard = await transcribe(session, model, state, spoken)
            messages.append({"role": "user", "content": heard})
            started, first, reply = time.monotonic(), None, ""
            stream = await api.chat.completions.create(
                model="gb-phone", messages=messages, stream=True, max_tokens=250,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    first = first or time.monotonic()
                    reply += delta
            ttft = (first - started) if first else None
            if index:  # the first turn includes any cold start
                ttfts.append(ttft or 0)
            messages.append({"role": "assistant", "content": reply.strip()})
            log.append({"turn": index, "spoken": spoken, "heard": heard, "reply": reply.strip(), "ttft_s": ttft})
            print(f"[{index:02d}] ttft={ttft:.2f}s heard={heard!r}\n     -> {reply.strip()}")
    summary = {"turns": len(log), "warm_ttft_p50_s": statistics.median(ttfts),
               "warm_ttft_p95_s": sorted(ttfts)[int(0.95 * (len(ttfts) - 1))], "empty_replies": sum(not r["reply"] for r in log)}
    (OUT / args.out).write_text(json.dumps({"url": args.url, "prompt": args.prompt, "summary": summary, "log": log}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
