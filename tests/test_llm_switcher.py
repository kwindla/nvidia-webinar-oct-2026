"""Regression checks for round-trip flush probes through the LLM switcher."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import PipelineFlushFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService

from llm_switcher import ConversationLLMSwitcher


class SwitcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_flush_completes_three_legs_without_duplicate_branch_output(self):
        services = {key: OpenAILLMService(api_key="test") for key in ("cloud", "local")}
        switcher = ConversationLLMSwitcher(services)
        switcher.push_frame = AsyncMock()
        probe = PipelineFlushFrame()
        for direction, returning in (
            (FrameDirection.DOWNSTREAM, False),
            (FrameDirection.UPSTREAM, False),
            (FrameDirection.DOWNSTREAM, True),
        ):
            probe.returning = returning
            for _ in services:  # Both branches can emit the same probe.
                await switcher._parallel_push_frame(probe, direction)
        self.assertEqual(switcher.push_frame.await_count, 3)

    async def test_selection_is_validated_and_does_not_switch_in_flight_reply(self):
        services = {key: OpenAILLMService(api_key="test") for key in ("cloud", "local")}
        switcher = ConversationLLMSwitcher(services)
        switcher.select("local")
        self.assertEqual(switcher.selected, "local")
        self.assertIs(switcher.strategy.active_service, services["cloud"])
        for invalid in (None, [], "unknown", "http://arbitrary-host"):
            with self.assertRaises(ValueError):
                switcher.select(invalid)
        self.assertEqual(switcher.selected, "local")
