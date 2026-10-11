"""Pipecat 1.12 Flux guard against late completion events after barge-in."""

from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService


class DemoFluxTTSService(DeepgramFluxTTSService):
    """Keep upstream streaming/prosody, but correlate completion by speech ID.

    Upstream attributes SpeechMetadata to whichever context is active on receipt.
    A completion queued before Interrupt can otherwise close the *next* context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._speech_context = None
        self._awaiting_interrupt_ack = False

    async def on_audio_context_interrupted(self, context_id):
        self._speech_context = None
        self._awaiting_interrupt_ack = self._transport_is_active()
        await super().on_audio_context_interrupted(context_id)

    async def _handle_message(self, msg):
        kind = msg.get("type")
        if kind == "SpeechInterrupted" or (
            kind == "Warning" and msg.get("code") == "NO_ACTIVE_SPEECH"
        ):
            self._awaiting_interrupt_ack = False
        elif kind == "SpeechStarted":
            context_id = self.get_active_audio_context_id()
            if not self._awaiting_interrupt_ack and context_id:
                self._speech_context = (msg.get("speech_id"), context_id)
        elif kind == "SpeechMetadata":
            context_id = self.get_active_audio_context_id()
            if (
                self._awaiting_interrupt_ack
                or not context_id
                or self._speech_context != (msg.get("speech_id"), context_id)
            ):
                return
            self._speech_context = None
        await super()._handle_message(msg)

    async def _handle_audio(self, audio):
        if (
            not self._awaiting_interrupt_ack
            and self._speech_context
            and self._speech_context[1] == self.get_active_audio_context_id()
        ):
            await super()._handle_audio(audio)

    async def _disconnect(self):
        await super()._disconnect()
        self._speech_context = None
        self._awaiting_interrupt_ack = False
