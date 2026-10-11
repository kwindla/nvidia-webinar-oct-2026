"""Live mirrored-ASR/Jev smoke test using deterministic speech boundaries."""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aiohttp
import soundfile as sf
from dotenv import load_dotenv
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from asr_comparison import ComparisonSession
from streaming_stt import NemoSpeechCppSTTService

REFERENCE = "United four fifty two turn left heading two seven zero descend and maintain five thousand feet"


async def main():
    load_dotenv(".env")
    audio, rate = sf.read("artifacts/synthetic-16k.wav", dtype="int16")
    async with aiohttp.ClientSession() as http:
        comparison = ComparisonSession(http)
        comparison.configure({"reference": REFERENCE, "context_judging": False})
        stt = NemoSpeechCppSTTService(session=http, comparison=comparison)
        worker = PipelineWorker(
            Pipeline([stt]),
            params=PipelineParams(audio_in_sample_rate=16000),
            enable_rtvi=False,
            enable_turn_tracking=False,
        )
        ready = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def started(worker, frame):
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)

        async def publish(snapshot):
            pass

        await comparison.start(publish)
        run = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(ready.wait(), 15)
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            pcm = audio.tobytes()
            for offset in range(0, len(pcm), 640):
                await worker.queue_frame(
                    InputAudioRawFrame(pcm[offset : offset + 640], rate, 1)
                )
                await asyncio.sleep(0.02)
            stop = VADUserStoppedSpeakingFrame()
            await worker.queue_frame(stop)
            # Exercise the stop-before-commit-handler ordering too.
            comparison.turn_stopped()
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                snapshot = comparison.snapshot()
                if snapshot["turns"] and snapshot["turns"][0]["judge"]["state"] in (
                    "done",
                    "unavailable",
                    "skipped",
                ):
                    break
                await asyncio.sleep(0.1)
            Path("artifacts/comparison-smoke.json").write_text(
                json.dumps(snapshot, indent=2) + "\n"
            )
            print(json.dumps(snapshot, indent=2))
            turn = snapshot["turns"][0]
            assert all(
                turn["sources"][s]["state"] == "final" for s in ("fine-tuned", "base")
            )
            assert turn["judge"]["state"] == "done", turn["judge"]
            assert (
                turn["sources"]["fine-tuned"]["final"]
                and turn["sources"]["base"]["final"]
            )
        finally:
            await comparison.close()
            await runner.cancel()
            await run


if __name__ == "__main__":
    asyncio.run(main())
