import asyncio
import time
import unittest

from asr_comparison import ComparisonSession, word_error_rate


class ComparisonTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        c = ComparisonSession(None)
        c.status = {s: "ready" for s in ("fine-tuned", "base")}
        c.configure({"reference": "descend five thousand feet", "context_judging": False})
        c.speech_started(time.time() - 1)
        return c

    def ack(self, c, source, text=None):
        if text is not None:
            c.event(source, {"type": "x.completed", "transcript": text})
        c.event(source, {"type": "input_audio_buffer.committed"})

    async def test_waits_for_ack_and_pairs_late_baseline_with_original_turn(self):
        c = self.make()
        c.reserve_commit()
        c.turn_stopped()
        c.event(
            "fine-tuned",
            {"type": "x.completed", "transcript": "descend five thousand feet"},
        )
        self.assertEqual(c.turns[1].judge["state"], "waiting")
        self.ack(c, "fine-tuned")
        c.speech_started()
        c.reserve_commit()
        c.turn_stopped()
        self.ack(c, "base", "descend nine thousand feet")
        self.assertEqual(c.turns[1].judge["state"], "pending")
        self.assertEqual(
            c.source_text(c.turns[1], "base")["final"], "descend nine thousand feet"
        )
        self.assertEqual(c.source_text(c.turns[2], "base")["final"], "")
        self.assertEqual(c.turns[2].reference, "")

    async def test_empty_ack_closes_and_clears_partial_without_final(self):
        c = self.make()
        c.reserve_commit()
        c.turn_stopped()
        c.event("base", {"type": "x.delta", "delta": "NOISE"})
        self.ack(c, "base")
        self.ack(c, "fine-tuned")
        self.assertEqual(c.source_text(c.turns[1], "base")["interim"], "")
        self.assertEqual(c.turns[1].judge["winner"], "identical")

    async def test_delayed_commit_after_stop_uses_vad_timestamp(self):
        c = self.make()
        timestamp = time.time()
        c.turn_stopped()
        c.reserve_commit(timestamp)
        self.ack(c, "fine-tuned", "one")
        self.ack(c, "base", "two")
        self.assertEqual(list(c.turns), [1])
        self.assertEqual(c.turns[1].judge["state"], "pending")

    async def test_multisegment_turn_and_one_shot_reference(self):
        c = self.make()
        for text in ["descend", "five thousand feet"]:
            c.reserve_commit()
            for s in ["fine-tuned", "base"]:
                self.ack(c, s, text)
        c.turn_stopped()
        self.assertEqual(
            c.source_text(c.turns[1], "base")["final"], "descend five thousand feet"
        )
        self.assertEqual(c.turns[1].judge["winner"], "identical")
        self.assertEqual(c.reference, "")
        self.assertEqual(c.speech_started().reference, "")

    async def test_protocol_error_and_unsolicited_ack_are_terminal(self):
        for event in [{"type": "error"}, {"type": "input_audio_buffer.committed"}]:
            c = self.make()
            c.event("base", event)
            self.assertEqual(c.status["base"], "unavailable")
            c.reserve_commit()
            c.turn_stopped()
            self.ack(c, "fine-tuned", "one")
            self.ack(c, "base", "two")
            self.assertEqual(c.turns[1].judge["state"], "unavailable")

    async def test_lag_limit_and_idle_audio(self):
        c = ComparisonSession(None)
        c.status["base"] = "ready"
        # Simulate consumer draining idle audio; lack of silence events is not lag.
        for _ in range(20):
            c.offer_audio(bytes(16000))
            c.audio_queue.get_nowait()
            c.queued_bytes = 0
        self.assertEqual(c.status["base"], "ready")
        c.speech_started()
        for _ in range(17):
            c.offer_audio(bytes(16000))
            if not c.audio_queue.empty():
                c.audio_queue.get_nowait()
                c.queued_bytes = 0
        self.assertEqual(c.status["base"], "ready")
        c.reserve_commit()
        self.ack(c, "base")
        self.assertEqual(c.status["base"], "ready")

    async def test_queue_overflow_and_shutdown(self):
        c = self.make()
        for _ in range(501):
            c.offer_commit()
        self.assertEqual(c.errors["base"], "baseline_queue_full")
        task = asyncio.create_task(c._deadlines())
        c.tasks = [task]
        await asyncio.wait_for(c.close(), 1)
        self.assertTrue(task.cancelled())
        self.assertEqual(c.audio_queue.qsize(), 0)

    async def test_reference_validation_and_wer(self):
        c = self.make()
        for bad in [
            {"reference": "x\n", "context_judging": False},
            {"reference": "x" * 501, "context_judging": False},
            {"reference": "x", "context_judging": "yes"},
        ]:
            with self.assertRaises(ValueError):
                c.configure(bad)
        self.assertEqual(word_error_rate("one two", "one nine"), 0.5)
        self.assertEqual(word_error_rate("one", "one two three"), 2)
        self.assertIsNone(word_error_rate("", "one"))

    async def test_disconnect_cancels_inflight_judge(self):
        waiting = asyncio.Event()

        async def blocked(turn, session):
            waiting.set()
            await asyncio.Event().wait()

        c = self.make()
        c.judge_call = blocked
        c.reserve_commit()
        c.turn_stopped()
        self.ack(c, "fine-tuned", "one")
        self.ack(c, "base", "two")
        c.tasks = [asyncio.create_task(c._judge_worker())]
        await asyncio.wait_for(waiting.wait(), 1)
        await asyncio.wait_for(c.close(), 1)
        self.assertEqual(c.turns[1].judge["state"], "pending")


if __name__ == "__main__":
    unittest.main()


class ReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    make = ComparisonTests.make
    ack = ComparisonTests.ack

    async def test_noise_text_cannot_create_turn_or_consume_reference(self):
        c = ComparisonSession(None)
        c.configure({"reference": "hello", "context_judging": False})
        c.event("fine-tuned", {"type": "x.delta", "delta": "hallucination"})
        self.assertEqual(c.turns, {})
        self.assertEqual(c.reference, "hello")
        self.assertIsNone(c.segments[1].turn)
        c.speech_started()
        self.assertEqual(c.segments[1].turn, 1)

    async def test_early_next_segment_does_not_poison_sealed_turn(self):
        c = self.make()
        c.reserve_commit()
        self.ack(c, "fine-tuned", "one")
        self.ack(c, "base", "one")
        c.event("fine-tuned", {"type": "x.delta", "delta": "noise"})
        c.turn_stopped()
        self.assertEqual(c.turns[1].segments, [1])
        self.assertEqual(c.turns[1].judge["winner"], "identical")
        c.speech_started()
        self.assertEqual(c.segments[2].turn, 2)
        self.assertFalse(c.disabled)

    async def test_blank_turn_rearms_reference_unless_user_changed_it(self):
        c = self.make()
        c.reserve_commit()
        c.turn_stopped()
        self.ack(c, "fine-tuned")
        self.ack(c, "base")
        self.assertEqual(c.reference, "descend five thousand feet")
        self.assertEqual(c.turns[1].wer, {"fine-tuned": 1.0, "base": 1.0})
        c.speech_started()
        c.reserve_commit()
        c.turn_stopped()
        c.configure({"reference": "new reference", "context_judging": False})
        self.ack(c, "fine-tuned")
        self.ack(c, "base")
        self.assertEqual(c.reference, "new reference")

    async def test_wer_survives_judge_error(self):
        async def broken(*args):
            raise ValueError("simulated judge failure")

        c = self.make()
        c.judge_call = broken
        c.reserve_commit()
        c.turn_stopped()
        self.ack(c, "fine-tuned", "descend five thousand feet")
        self.ack(c, "base", "descend nine thousand feet")
        c.tasks = [asyncio.create_task(c._judge_worker())]
        await asyncio.sleep(0.01)
        self.assertEqual(c.turns[1].judge["state"], "unavailable")
        self.assertEqual(c.turns[1].wer, {"fine-tuned": 0, "base": 0.25})
        await c.close()

    async def test_publisher_retries_without_blaming_asr(self):
        c = self.make()
        delivered = asyncio.Event()
        calls = 0

        async def flaky(snapshot):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("temporary delivery error")
            delivered.set()

        task = asyncio.create_task(c._publisher(flaky))
        c.tasks = [task]
        c.set_publish_ready()
        c.touch()
        await asyncio.wait_for(delivered.wait(), 2)
        self.assertEqual(c.status["base"], "ready")
        await c.close()
