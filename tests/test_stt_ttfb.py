import unittest
from unittest import mock

from pipecat.services.stt_service import STTService

from streaming_stt import NemoSpeechCppSTTService


class SttTtfbStartTest(unittest.IsolatedAsyncioTestCase):
    async def test_ttfb_starts_at_vad_stop_not_estimated_speech_end(self):
        stt = NemoSpeechCppSTTService(session=None)
        stt._vad_stop_secs = 0.4
        with mock.patch.object(STTService, "start_ttfb_metrics") as base:
            # Pipecat passes VAD stop timestamp minus stop_secs.
            await stt.start_ttfb_metrics(start_time=100.0 - 0.4)
            await stt.start_ttfb_metrics()
        self.assertAlmostEqual(base.await_args_list[0].kwargs["start_time"], 100.0)
        self.assertIsNone(base.await_args_list[1].kwargs["start_time"])


if __name__ == "__main__":
    unittest.main()
