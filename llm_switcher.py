"""Select an LLM at response boundaries, preserving the shared context."""

from collections import deque

from pipecat.frames.frames import LLMContextFrame, ManuallySwitchServiceFrame, PipelineFlushFrame
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.processors.frame_processor import FrameDirection


class ConversationLLMSwitcher(LLMSwitcher):
    def __init__(self, services, selected="cloud"):
        self.models = services
        self.selected = selected if selected in services else next(iter(services))
        self._flush_legs = deque(maxlen=256)
        super().__init__(llms=list(services.values()))

    async def _parallel_push_frame(self, frame, direction):
        # Pipecat 1.12 deduplicates parallel frames solely by id, but its flush
        # probe reuses that id for downstream → upstream → downstream. Allow
        # each leg once, while still deduplicating copies from parallel branches.
        if isinstance(frame, PipelineFlushFrame):
            leg = (frame.id, direction, frame.returning)
            if leg in self._flush_legs:
                return
            self._flush_legs.append(leg)
            self._seen_ids.discard(frame.id)
        await super()._parallel_push_frame(frame, direction)

    def select(self, model):
        if not isinstance(model, str) or model not in self.models:
            raise ValueError(f"Choose one of: {', '.join(self.models)}.")
        self.selected = model

    async def process_frame(self, frame, direction):
        # Do not cut off an in-flight completion or its queued speech. The next
        # context frame carries the existing conversation to the selected LLM.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            service = self.models[self.selected]
            if self.strategy.active_service is not service:
                await super().process_frame(ManuallySwitchServiceFrame(service=service), direction)
        await super().process_frame(frame, direction)
