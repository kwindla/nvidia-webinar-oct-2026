"""SmallWebRTC voice demo: Nemotron ASR → selectable cloud/local/tuned LLM → Deepgram Flux TTS."""

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from asr_comparison import ComparisonSession
from flux_tts import DemoFluxTTSService
from llm_switcher import ConversationLLMSwitcher
from model_choices import model_catalog, register_model_routes, tuned_api_key
from streaming_stt import NemoSpeechCppSTTService
from tutor_prompts import SHORT13_PROMPT, TUTOR_PROMPT, VOICE_PROMPT

load_dotenv(Path(__file__).with_name(".env"))

GREETING = "Let’s practice aviation radio calls. Say an aviation phrase, and I’ll briefly explain what it means."


async def queue_greeting(worker):
    # Pipecat TTS emits spoken text/RTVI and commits it through the assistant aggregator.
    # Do not also add a context message here: that would duplicate the greeting.
    await worker.queue_frame(TTSSpeakFrame(GREETING, append_to_context=True))


async def bot(runner_args: RunnerArguments):
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise TypeError("Run this demo with -t webrtc")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the project .env")
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_key:
        raise RuntimeError("Set DEEPGRAM_API_KEY in the project .env")
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    context = LLMContext()
    async with aiohttp.ClientSession() as session:
        comparison = ComparisonSession(
            session,
            base_url=os.getenv("BASE_ASR_URL", "http://127.0.0.1:8082"),
            context_getter=context.get_messages,
        )
        stt = NemoSpeechCppSTTService(
            comparison=comparison,
            name="nemotron-asr",
            session=session,
            url=os.getenv("NVIDIA_ASR_URL", "http://127.0.0.1:8081"),
        )
        cloud_llm = OpenAILLMService(
            name="cloud-llm",
            api_key=api_key,
            settings=OpenAILLMService.Settings(
                model="gpt-5.6-terra",
                system_instruction=VOICE_PROMPT,
                extra={"reasoning_effort": "medium"},
                max_completion_tokens=4096,
            ),
        )
        catalog = await model_catalog(session)
        names = catalog.names
        llms = {"cloud": cloud_llm}
        if "local" in names:
            llms["local"] = OpenAILLMService(
                name="local-llm",
                base_url=os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1"),
                # The SDK requires a value; the local server requires no credential.
                api_key="not-required",
                settings=OpenAILLMService.Settings(
                    model=os.getenv("LOCAL_LLM_MODEL", "gb-phone"),
                    system_instruction=VOICE_PROMPT,
                    max_completion_tokens=250,
                ),
            )
        if "tuned" in names:
            tuned_url = os.environ["TUNED_LLM_URL"]
            # The ATC tutor fine-tune (local vLLM or Modal) with the short prompt it was trained on:
            # v1.3 models use the v1.3 prompt, v2/v3 the original one.
            tuned_prompt = SHORT13_PROMPT if os.getenv("TUNED_PROMPT", "v2") == "v13" else TUTOR_PROMPT
            llms["tuned"] = OpenAILLMService(
                name="tuned-llm",
                base_url=tuned_url,
                api_key=tuned_api_key(tuned_url),
                settings=OpenAILLMService.Settings(
                    model="gb-phone",
                    system_instruction=tuned_prompt,
                    max_completion_tokens=250,
                    # Nemotron chat-template option; not sent to the OpenAI arm.
                    extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                ),
            )
        body = runner_args.body if isinstance(runner_args.body, dict) else {}
        selected, fallback = catalog.initial_selection(body.get("model"))
        llm = ConversationLLMSwitcher(llms, selected=selected)
        tts = DemoFluxTTSService(
            name="deepgram-flux-tts",
            api_key=deepgram_key,
            settings=DemoFluxTTSService.Settings(
                voice=os.getenv("DEEPGRAM_TTS_VOICE", "flux-heather-en"),
            ),
        )
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=LocalSmartTurnAnalyzerV3(),
                        ),
                    ]
                ),
            ),
        )
        worker = PipelineWorker(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    aggregators.user(),
                    llm,
                    tts,
                    transport.output(),
                    aggregators.assistant(),
                ]
            ),
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                enable_metrics=True,
                enable_usage_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        @worker.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            nonlocal fallback
            if message.type in ("get-comparison", "set-comparison"):
                if message.type == "set-comparison":
                    try:
                        comparison.configure(message.data)
                    except ValueError as error:
                        await rtvi.send_error_response(message, str(error))
                        return
                await rtvi.send_server_response(message, comparison.snapshot())
                return
            if message.type == "set-model":
                try:
                    data = message.data if isinstance(message.data, dict) else {}
                    llm.select(data.get("model"))
                except ValueError as error:
                    await rtvi.send_error_response(message, str(error))
                    return
                fallback = None
                logger.info("Selected {} LLM for the next response", llm.selected)
            elif message.type != "get-model":
                return
            await rtvi.send_server_response(
                message,
                {"model": llm.selected, "available": list(llm.models),
                 "names": names, "fallback": fallback},
            )

        greeted = False

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, client):
            nonlocal greeted
            comparison.set_publish_ready()
            if not greeted:
                greeted = True
                logger.info("Browser connected; speaking fixed greeting")
                await queue_greeting(worker)

        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(transport, client):
            await comparison.close()
            await worker.cancel()

        @worker.event_handler("on_pipeline_error")
        async def on_error(worker, frame):
            logger.error("Voice pipeline error: {}", frame.error)

        runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
        await runner.add_workers(worker)
        await comparison.start(worker.rtvi.send_server_message)
        try:
            await runner.run()
        finally:
            await comparison.close()


if __name__ == "__main__":
    from pipecat.runner.run import app, main

    register_model_routes(app)

    from cloudflare_turn import configure_runner_ice

    if os.getenv("CLOUDFLARE_TURN_KEY_ID") and any(
        arg == "--ice-servers" or arg.startswith("--ice-servers=")
        for arg in sys.argv[1:]
    ):
        raise SystemExit("Remove --ice-servers to use Cloudflare TURN from .env")
    asyncio.run(configure_runner_ice())
    main()
