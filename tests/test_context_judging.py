import unittest

from asr_comparison import ComparisonSession, recent_history


class ContextTests(unittest.TestCase):
    def test_context_is_bounded_filtered_and_frozen_at_speech_start(self):
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(14)
        ]
        messages += [
            {"role": "system", "content": "private instruction"},
            {"role": "tool", "content": "tool payload"},
        ]
        c = ComparisonSession(None, context_getter=lambda: messages)
        turn = c.speech_started()
        self.assertTrue(turn.context_judging)
        self.assertEqual(
            [m["text"] for m in turn.history], [str(i) for i in range(4, 14)]
        )
        messages[4]["content"] = "changed"
        messages.append({"role": "assistant", "content": "response to current ASR"})
        self.assertEqual(turn.history[0]["text"], "4")
        self.assertNotIn("response to current ASR", str(turn.history))

    def test_multimodal_only_text_and_truncation(self):
        history = recent_history(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "secret"},
                        {"type": "text", "text": "x" * 3000},
                    ],
                }
            ]
        )
        self.assertEqual(len(history[0]["text"]), 2000)
        self.assertTrue(history[0]["truncated"])
        self.assertNotIn("secret", str(history))

    def test_default_judges_different_transcripts_without_reference(self):
        c = ComparisonSession(None)
        c.status = {s: "ready" for s in ("fine-tuned", "base")}
        turn = c.speech_started()
        c.reserve_commit()
        c.turn_stopped()
        for source, text in [("fine-tuned", "heading"), ("base", "wedding")]:
            c.event(source, {"type": "x.completed", "transcript": text})
            c.event(source, {"type": "input_audio_buffer.committed"})
        self.assertEqual(turn.judge["state"], "pending")
        self.assertEqual(c.judge_queue.get_nowait(), turn.id)
