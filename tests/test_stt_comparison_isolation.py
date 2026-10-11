import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

from streaming_stt import NemoSpeechCppSTTService


class Shadow:
    def __init__(self, broken=False):
        self.calls = []
        self.broken = broken

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if self.broken and name != "fail":
                raise RuntimeError("comparison bug")

        return call


class FakeSocket:
    def __init__(self):
        self.calls = []

    async def send_bytes(self, audio):
        self.calls.append(("audio", audio))

    async def send_json(self, data):
        self.calls.append(("commit", data))

    async def __aiter__(self):
        for event in [
            {"type": "x.delta", "delta": "HELLO"},
            {"type": "x.completed", "transcript": "hello"},
            {"type": "input_audio_buffer.committed"},
        ]:
            yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mirror_order_and_commit_survive_hook_exceptions(self):
        for broken in [False, True]:
            shadow = Shadow(broken)
            stt = NemoSpeechCppSTTService(session=None, comparison=shadow)
            stt.ws = FakeSocket()
            with patch.object(
                STTService, "process_frame", new_callable=AsyncMock
            ) as parent:
                await stt.process_frame(
                    VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM
                )
                for data in [b"first", b"second"]:
                    async for _ in stt.run_stt(data):
                        pass
                await stt.process_frame(
                    VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM
                )
                await stt.process_frame(
                    UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM
                )
                self.assertEqual(parent.await_count, 3)
            self.assertEqual([k for k, _ in stt.ws.calls], ["audio", "audio", "commit"])
            self.assertEqual(
                [args[0] for name, args in shadow.calls if name == "offer_audio"],
                [b"first", b"second"],
            )
            names = [name for name, _ in shadow.calls]
            self.assertLess(names.index("reserve_commit"), names.index("offer_commit"))
            self.assertLess(names.index("offer_commit"), names.index("turn_stopped"))

    async def test_shadow_receiver_exception_cannot_suppress_primary_transcripts(self):
        stt = NemoSpeechCppSTTService(session=None, comparison=Shadow(True))
        stt.ws = FakeSocket()
        with (
            patch.object(stt, "push_frame", new_callable=AsyncMock) as push,
            patch.object(stt, "push_error", new_callable=AsyncMock),
        ):
            await stt._receive()
        frames = [call.args[0] for call in push.await_args_list]
        self.assertEqual(len(frames), 2)
        self.assertIsInstance(frames[0], InterimTranscriptionFrame)
        self.assertIsInstance(frames[1], TranscriptionFrame)
        self.assertEqual([frame.text for frame in frames], ["hello", "hello"])
        self.assertEqual(stt._partial_text, "")
