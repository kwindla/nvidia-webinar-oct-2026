import unittest
from unittest.mock import AsyncMock, patch

from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService

from flux_tts import DemoFluxTTSService


class FluxTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_completion_cannot_close_new_context(self):
        tts = DemoFluxTTSService(api_key="test")
        active = "old"
        tts.get_active_audio_context_id = lambda: active
        tts._transport_is_active = lambda: True
        with (
            patch.object(
                DeepgramFluxTTSService, "_handle_message", new_callable=AsyncMock
            ) as handle,
            patch.object(
                DeepgramFluxTTSService,
                "on_audio_context_interrupted",
                new_callable=AsyncMock,
            ),
        ):
            await tts._handle_message(
                {"type": "SpeechStarted", "speech_id": "old-speech"}
            )
            await tts.on_audio_context_interrupted("old")
            active = "new"
            handle.reset_mock()
            await tts._handle_message(
                {"type": "SpeechMetadata", "speech_id": "old-speech"}
            )
            handle.assert_not_awaited()
            await tts._handle_message(
                {"type": "SpeechInterrupted", "speech_id": "old-speech"}
            )
            await tts._handle_message(
                {"type": "SpeechStarted", "speech_id": "new-speech"}
            )
            handle.reset_mock()
            await tts._handle_message(
                {"type": "SpeechMetadata", "speech_id": "old-speech"}
            )
            handle.assert_not_awaited()
            await tts._handle_message(
                {"type": "SpeechMetadata", "speech_id": "new-speech"}
            )
            handle.assert_awaited_once()

    async def test_interrupted_audio_is_discarded_until_ack_and_new_start(self):
        tts = DemoFluxTTSService(api_key="test")
        tts.get_active_audio_context_id = lambda: "new"
        tts._awaiting_interrupt_ack = True
        with (
            patch.object(
                DeepgramFluxTTSService, "_handle_audio", new_callable=AsyncMock
            ) as audio,
            patch.object(
                DeepgramFluxTTSService, "_handle_message", new_callable=AsyncMock
            ),
        ):
            await tts._handle_audio(b"old")
            await tts._handle_message(
                {"type": "SpeechStarted", "speech_id": "late-old"}
            )
            await tts._handle_audio(b"old")
            audio.assert_not_awaited()
            await tts._handle_message({"type": "Warning", "code": "NO_ACTIVE_SPEECH"})
            await tts._handle_audio(b"old")
            audio.assert_not_awaited()
            await tts._handle_message({"type": "SpeechStarted", "speech_id": "new"})
            await tts._handle_audio(b"new")
            audio.assert_awaited_once_with(b"new")
