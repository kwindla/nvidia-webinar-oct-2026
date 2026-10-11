"""Send synthetic speech through Pipecat to the local Nemotron server."""
import argparse
import asyncio
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

import aiohttp
import soundfile as sf
from pipecat.frames.frames import (
    EndFrame, InputAudioRawFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from streaming_stt import NemoSpeechCppSTTService

EXPECTED = ('United four fifty two, turn left heading two seven zero. '
            'Descend and maintain five thousand feet.')


class TranscriptCollector(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.done = asyncio.Event()
        self.transcripts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self.transcripts.append({'text': frame.text, 'finalized': frame.finalized})
            self.done.set()
        await self.push_frame(frame, direction)


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


async def main(args):
    audio, rate = sf.read(args.audio, dtype='int16')
    if rate != 16000 or audio.ndim != 1:
        raise ValueError('Smoke input must be mono 16 kHz WAV')
    pcm = audio.astype('<i2').tobytes()
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{args.url}/health') as response:
            response.raise_for_status()
            health = await response.json()
        stt = NemoSpeechCppSTTService(session=session, url=args.url)
        collector = TranscriptCollector()
        worker = PipelineWorker(
            Pipeline([stt, collector]),
            params=PipelineParams(audio_in_sample_rate=16000),
            enable_rtvi=False, enable_turn_tracking=False,
        )
        started = asyncio.Event()
        errors = []

        @worker.event_handler('on_pipeline_started')
        async def on_started(worker, frame):
            started.set()

        @worker.event_handler('on_pipeline_error')
        async def on_error(worker, frame):
            errors.append(frame.error)
            collector.done.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        run = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(started.wait(), 15)
            # Deterministic speech boundaries isolate ASR from VAD accuracy.
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            for offset in range(0, len(pcm), 640):  # 20 ms of mono PCM16
                await worker.queue_frame(InputAudioRawFrame(pcm[offset:offset + 640], rate, 1))
                await asyncio.sleep(0.02)
            finish = time.perf_counter()
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.wait_for(collector.done.wait(), 120)
            latency = time.perf_counter() - finish
            await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run, 15)
        finally:
            if not run.done():
                await runner.cancel()
                await run
        if errors:
            raise RuntimeError(f'Pipecat errors: {errors}')
        actual = ' '.join(t['text'] for t in collector.transcripts)
        passed = (words(actual) == words(args.expected)
                  and all(t['finalized'] for t in collector.transcripts)
                  and stt.partial_count > 0)
        report = {
            'passed': passed, 'expected': args.expected,
            'transcripts': collector.transcripts, 'audio_file': args.audio,
            'audio_seconds': len(audio) / rate,
            'speech_end_to_transcript_seconds': latency,
            'server_response': stt.last_response, 'server': health,
            'streaming_partial_count': stt.partial_count,
            'versions': {p: version(p) for p in ['pipecat-ai', 'pocket-tts', 'torch']},
        }
        Path('artifacts').mkdir(exist_ok=True)
        Path('artifacts/smoke-result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        if not passed:
            raise AssertionError('Transcript differs from expected normalized words; see report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', default='artifacts/synthetic-16k.wav')
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--expected', default=EXPECTED)
    asyncio.run(main(parser.parse_args()))
