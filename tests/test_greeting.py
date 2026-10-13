import hashlib
import os
from contextlib import ExitStack
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import AggregationType, TTSStartedFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver
from pipecat.processors.frame_processor import FrameDirection

import bot as app_bot
from bot import GREETING, queue_greeting
from model_choices import ModelCatalog
from pipecat.runner.types import SmallWebRTCRunnerArguments
from flux_tts import DemoFluxTTSService
from tutor_prompts import DEMO_PROMPT, VOICE_PROMPT, FULL_PROMPT, FULL13_PROMPT, TRAINED_PROMPT_SHA256_PREFIX


class GreetingTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_tts_frame_and_spoken_context_for_every_model(self):
        for model in ('cloud', 'local', 'tuned'):
            with self.subTest(model=model):
                worker = SimpleNamespace(queue_frame=AsyncMock())
                await queue_greeting(worker)
                worker.queue_frame.assert_awaited_once()
                frame = worker.queue_frame.call_args.args[0]
                self.assertIsInstance(frame, TTSSpeakFrame)
                self.assertEqual(frame.text, 'Let’s practice aviation radio calls. Say an aviation phrase, and I’ll briefly explain what it means.')
                self.assertTrue(frame.append_to_context)

                # Exercise the real Pipecat TTSSpeakFrame branch, without opening a TTS socket.
                tts = DemoFluxTTSService(api_key='test')
                tts._push_tts_frames = AsyncMock()
                tts.on_turn_context_created = AsyncMock()
                tts.on_turn_context_completed = AsyncMock()
                tts._push_sequencer_frames = AsyncMock()
                tts._maybe_pause_frame_processing = AsyncMock()
                await tts.process_frame(frame, FrameDirection.DOWNSTREAM)
                call = tts._push_tts_frames.call_args
                self.assertEqual(call.args[0].text, GREETING)
                self.assertTrue(call.kwargs['append_tts_text_to_context'])
                self.assertTrue(call.kwargs['push_assistant_aggregation'])

                # Playback-generated frames follow output transport into the real aggregator.
                context = LLMContext()
                aggregator = LLMAssistantAggregator(context)
                aggregator.push_frame = AsyncMock()
                aggregator.broadcast_frame = AsyncMock()
                spoken = TTSTextFrame(GREETING, AggregationType.SENTENCE)
                spoken.will_be_spoken = True
                await aggregator._handle_tts_started(TTSStartedFrame())
                await aggregator._handle_text(spoken)
                await aggregator._handle_push_aggregation()
                await aggregator._handle_push_aggregation()  # a duplicate stop must not duplicate history
                self.assertEqual(context.get_messages(), [{'role': 'assistant', 'content': GREETING}])

                observer = RTVIObserver()
                observer.send_rtvi_message = AsyncMock()
                await observer._send_aggregated_llm_text(spoken)
                messages = [c.args[0].model_dump() for c in observer.send_rtvi_message.call_args_list]
                self.assertTrue(any(m['type'] == 'bot-output' and m['data']['text'] == GREETING for m in messages))

    async def test_real_connect_handler_greets_once_and_reports_fallback(self):
        for requested in ('cloud', 'local', 'tuned'):
            with self.subTest(requested=requested), ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {
                    'OPENAI_API_KEY': 'test', 'DEEPGRAM_API_KEY': 'test',
                    'TUNED_LLM_URL': 'http://localhost:8083/v1',
                }, clear=True))
                catalog = ModelCatalog({'cloud': 'Cloud', 'tuned': 'Tuned'}, {'local': 'The server timed out.'})
                stack.enter_context(patch.object(app_bot, 'model_catalog', new=AsyncMock(return_value=catalog)))
                handlers = {}
                transport = MagicMock()
                transport.event_handler.side_effect = lambda event: lambda callback: handlers.setdefault(event, callback)
                stack.enter_context(patch.object(app_bot, 'SmallWebRTCTransport', return_value=transport))
                worker = MagicMock()
                worker.queue_frame = AsyncMock()
                worker.rtvi.event_handler.side_effect = lambda event: lambda callback: handlers.setdefault(event, callback)
                worker.event_handler.side_effect = lambda event: lambda callback: callback
                stack.enter_context(patch.object(app_bot, 'PipelineWorker', return_value=worker))
                for name in ('Pipeline', 'NemoSpeechCppSTTService', 'OpenAILLMService', 'DemoFluxTTSService',
                             'LLMContextAggregatorPair', 'SileroVADAnalyzer', 'LocalSmartTurnAnalyzerV3'):
                    stack.enter_context(patch.object(app_bot, name))
                comparison = MagicMock()
                comparison.start = AsyncMock()
                comparison.close = AsyncMock()
                stack.enter_context(patch.object(app_bot, 'ComparisonSession', return_value=comparison))
                stack.enter_context(patch.object(app_bot, 'ConversationLLMSwitcher', side_effect=lambda services, selected: SimpleNamespace(models=services, selected=selected)))
                rtvi = SimpleNamespace(send_server_response=AsyncMock())

                async def run():
                    await handlers['on_client_connected'](transport, None)
                    await handlers['on_client_connected'](transport, None)
                    await handlers['on_client_message'](rtvi, SimpleNamespace(type='get-model'))

                runner = SimpleNamespace(add_workers=AsyncMock(), run=run)
                stack.enter_context(patch.object(app_bot, 'WorkerRunner', return_value=runner))
                await app_bot.bot(SmallWebRTCRunnerArguments(webrtc_connection=MagicMock(), body={'model': requested}))
                worker.queue_frame.assert_awaited_once()
                self.assertIsInstance(worker.queue_frame.call_args.args[0], TTSSpeakFrame)
                self.assertEqual(worker.queue_frame.call_args.args[0].text, GREETING)
                result = rtvi.send_server_response.call_args.args[1]
                self.assertEqual(result['model'], 'cloud' if requested == 'local' else requested)
                if requested == 'local':
                    self.assertEqual(result['fallback']['reason'], 'The server timed out.')
                else:
                    self.assertIsNone(result['fallback'])

    def test_live_prompt_removed_greeting_historical_prompts_frozen(self):
        for phrase in ('At session start', 'If an actual user message is present'):
            self.assertNotIn(phrase, DEMO_PROMPT)
            self.assertNotIn(phrase, VOICE_PROMPT)
        self.assertEqual(hashlib.sha256(FULL_PROMPT.encode()).hexdigest()[:16], TRAINED_PROMPT_SHA256_PREFIX['full'])
        # Immutable fixture identities, including the v1.3 full prompt used by bench13/export13.
        self.assertEqual(hashlib.sha256(FULL13_PROMPT.encode()).hexdigest(), '432977f48477fea40b060b8441986b67b701d27545973cc74da1e5b36fb99a64')
