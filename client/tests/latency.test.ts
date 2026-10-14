import { test } from "node:test"
import assert from "node:assert/strict"
import {
  appendLatency,
  headline,
  latencyCeiling,
  WINDOW_SIZE,
  type LatencyHistory,
} from "../src/lib/latency.ts"

test("uses reported units and separates processors/models across switches", () => {
  const history = appendLatency(
    {},
    {
      ttfb: [
        { processor: "nemotron-asr", value: 0.2 },
        { processor: "cloud-llm", model: "gpt-6-luna", value: 0.3 },
      ],
      ttfat: [
        { processor: "cloud-llm", model: "gpt-6-luna", ttfat: 0.4, ttfb: 0.3 },
      ],
      ttfa: [{ processor: "pocket-tts", ttfa: 0.05, ttfb: 0.03 }],
    }
  )
  // TTS TTFA is not recorded; only its TTFB is shown.
  assert.equal(Object.keys(history).length, 3)
  assert.deepEqual(
    Object.values(history).map((s) => [s.kind, s.points]),
    [
      ["ttfs", [200]],
      ["ttfb", [300]],
      ["ttft", [400]],
    ]
  )
  const next = appendLatency(history, {
    ttfat: [{ processor: "local-llm", model: "gb-phone", ttfat: 0.02 }],
  })
  assert.equal(Object.keys(next).length, 4)
  assert.equal(Object.keys(history).length, 3)
  assert.equal(latencyCeiling(Object.values(next)), 500)
})
test("caps history, preserves measured zero, rejects invalid samples", () => {
  let history: LatencyHistory = {}
  for (let i = 0; i < WINDOW_SIZE + 5; i++)
    history = appendLatency(history, {
      ttfb: [{ processor: "x", value: i / 1000 }],
    })
  assert.equal(Object.values(history)[0].points.length, WINDOW_SIZE)
  assert.equal(Object.values(history)[0].points[0], 5)
  const invalid = appendLatency(history, {
    ttfb: [
      null,
      { processor: "x", value: -1 },
      { processor: "x", value: NaN },
      { processor: "x", value: "2" },
    ],
  })
  assert.equal(invalid, history)
  const zero = appendLatency({}, { ttfb: [{ processor: "x", value: 0 }] })
  assert.deepEqual(Object.values(zero)[0].points, [0])
})

test("shows one headline metric per service: LLM TTFT, ASR TTFS, else TTFB", () => {
  const history = appendLatency(
    {},
    {
      ttfb: [
        { processor: "nemotron-asr", value: 0.04 },
        { processor: "tuned-llm", value: 0.05 },
        { processor: "deepgram-flux-tts", value: 0.12 },
      ],
      ttfat: [{ processor: "tuned-llm", ttfat: 0.06 }],
      ttfa: [{ processor: "deepgram-flux-tts", ttfa: 0.2 }],
    }
  )
  const byProcessor = (name: string) =>
    headline(Object.values(history).filter((s) => s.processor === name))?.kind
  assert.equal(byProcessor("tuned-llm"), "ttft")
  assert.equal(byProcessor("nemotron-asr"), "ttfs")
  assert.equal(byProcessor("deepgram-flux-tts"), "ttfb")
  assert.equal(headline([]), undefined)
})
