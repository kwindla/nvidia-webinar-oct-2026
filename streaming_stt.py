"""Pipecat client for NVIDIA NeMo-Speech.cpp's realtime WebSocket protocol."""

import asyncio
import json

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from asr_comparison import SESSION_SETTINGS, append_partial


class NemoSpeechCppSTTService(STTService):
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        url="http://127.0.0.1:8081",
        comparison=None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=16000,
            settings=STTSettings(
                model="atc02-nemotron-asr-finetuned",
                language="en-US",
            ),
            **kwargs,
        )
        self.comparison = comparison
        self._send_lock = asyncio.Lock()
        self.session = session
        self.url = url.rstrip("/")
        self.ws = None
        self.receiver = None
        self.last_response = None
        self.partial_count = 0
        self._partial_text = ""

    def _compare(self, method, *args):
        """Auxiliary hooks must never prevent primary audio, commits, or transcripts."""
        if self.comparison is None:
            return
        try:
            getattr(self.comparison, method)(*args)
        except Exception:  # noqa: BLE001 - hard isolation from auxiliary bugs
            try:
                self.comparison.fail("fine-tuned", "comparison_internal_error")
            except Exception:  # noqa: BLE001 - even broken error reporting cannot kill STT
                logger.error(
                    "Comparison error reporting failed; detached auxiliary hooks"
                )
                self.comparison = None

    def can_generate_metrics(self):
        return True

    async def start(self, frame):
        await super().start(frame)
        self.ws = await self.session.ws_connect(
            f"{self.url}/v1/audio/transcriptions/realtime",
            heartbeat=20,
            timeout=aiohttp.ClientWSTimeout(ws_close=1),
        )
        created = await self.ws.receive_json(timeout=15)
        if created.get("type") != "session.created":
            raise RuntimeError(f"Unexpected server greeting: {created}")
        await self.ws.send_json({"type": "session.update", "session": SESSION_SETTINGS})
        updated = await self.ws.receive_json(timeout=15)
        if (
            updated.get("type") != "session.updated"
            or updated.get("session") != SESSION_SETTINGS
        ):
            raise RuntimeError("Failed to configure ASR stream")
        if self.comparison:
            self._compare("ready", "fine-tuned")
        self.receiver = self.create_task(self._receive())

    async def _receive(self):
        try:
            async for message in self.ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    kind = event.get("type", "")
                    if self.comparison:
                        self._compare("event", "fine-tuned", event)
                    if kind.endswith(".delta"):
                        if not event.get("delta"):
                            continue
                        self.partial_count += 1
                        self._partial_text = append_partial(
                            self._partial_text, event["delta"]
                        )
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                self._partial_text.lower(),
                                self._user_id,
                                time_now_iso8601(),
                            )
                        )
                    elif kind.endswith(".completed"):
                        self.last_response = event
                        self._partial_text = ""
                        if not event.get("transcript", "").strip():
                            continue
                        # The fine-tuned ASR emits uppercase finals; the tutor and
                        # the interim transcripts use lowercase spoken text.
                        await self.push_frame(
                            TranscriptionFrame(
                                event["transcript"].lower(),
                                self._user_id,
                                time_now_iso8601(),
                                finalized=True,
                            )
                        )
                    elif kind == "input_audio_buffer.committed":
                        self._partial_text = ""
                    elif kind == "error":
                        raise RuntimeError(str(event))
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error: {self.ws.exception()}")
            if self.comparison:
                self._compare("fail", "fine-tuned", "primary_connection_closed")
            await self.push_error("ASR WebSocket closed unexpectedly", fatal=True)
        except Exception:  # noqa: BLE001 - failure boundary; do not expose server payloads
            if self.comparison:
                self._compare("fail", "fine-tuned", "primary_connection_failed")
            await self.push_error("Primary ASR connection failed", fatal=True)

    async def run_stt(self, audio):
        async with self._send_lock:
            try:
                await self.ws.send_bytes(audio)
                if self.comparison:
                    self._compare("offer_audio", audio)
            except Exception:
                if self.comparison:
                    self._compare("fail", "fine-tuned", "primary_send_failed")
                raise
        # STTService requires an async generator; results arrive in _receive.
        if False:
            yield

    async def process_frame(self, frame, direction):
        if isinstance(frame, UserStoppedSpeakingFrame) and self.comparison:
            self._compare("turn_stopped")
        if isinstance(frame, VADUserStartedSpeakingFrame) and self.comparison:
            self._compare("speech_started", frame.timestamp)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            async with self._send_lock:
                if self.comparison:
                    self._compare("reserve_commit", frame.timestamp)
                await super().process_frame(frame, direction)
                try:
                    await self.ws.send_json({"type": "input_audio_buffer.commit"})
                    if self.comparison:
                        self._compare("offer_commit")
                except Exception:
                    if self.comparison:
                        self._compare("fail", "fine-tuned", "primary_commit_failed")
                    raise
        else:
            await super().process_frame(frame, direction)

    async def _close(self):
        self._compare("stop")
        if self.receiver:
            await self.cancel_task(self.receiver)
            self.receiver = None
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def stop(self, frame):
        await self._close()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._close()
        await super().cancel(frame)
