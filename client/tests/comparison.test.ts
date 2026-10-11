import { test } from "node:test"
import assert from "node:assert/strict"
import { acceptComparison, parseComparison } from "../src/lib/comparison.ts"
const snapshot = {
  type: "asr-comparison",
  version: 1,
  session_id: "session-a",
  revision: 1,
  status: { "fine-tuned": "ready", base: "ready" },
  errors: { "fine-tuned": null, base: null },
  models: { "fine-tuned": "ea", base: "3.5" },
  reference: "",
  context_judging: false,
  turns: [
    {
      id: 1,
      closed: true,
      reference: "one",
      sources: {
        "fine-tuned": {
          source: "fine-tuned",
          model: "ea",
          final: "one",
          interim: "",
          state: "final",
        },
        base: {
          source: "base",
          model: "3.5",
          final: "two",
          interim: "",
          state: "final",
        },
      },
      judge: {
        source: "jev",
        state: "done",
        winner: "fine-tuned",
        confidence: 0.9,
        mode: "reference",
      },
    },
  ],
}
test("accepts source-tagged snapshots and rejects stale/cross-session state", () => {
  const first = acceptComparison(null, snapshot)
  assert.ok(first)
  assert.equal(acceptComparison(first, { ...snapshot, revision: 0 }), first)
  assert.equal(
    acceptComparison(first, {
      ...snapshot,
      session_id: "session-b",
      revision: 2,
    }),
    first
  )
  assert.equal(
    acceptComparison(first, { ...snapshot, revision: 2 })?.revision,
    2
  )
})
test("rejects malformed transcripts, verdicts, and oversized histories", () => {
  assert.equal(
    parseComparison({ ...snapshot, turns: Array(21).fill(snapshot.turns[0]) }),
    null
  )
  assert.equal(
    parseComparison({
      ...snapshot,
      turns: [
        {
          ...snapshot.turns[0],
          judge: { source: "jev", state: "done", confidence: NaN },
        },
      ],
    }),
    null
  )
  assert.equal(
    parseComparison({
      ...snapshot,
      turns: [{ ...snapshot.turns[0], sources: {} }],
    }),
    null
  )
  assert.equal(parseComparison({ type: "other" }), null)
})

test("turn verdict labels separate preference, agreement, and uncertainty", async () => {
  const { judgmentLabel } = await import("../src/lib/comparison.ts")
  const labels = {
    "fine-tuned": "Fine-tuned better",
    base: "Base better",
    tie: "Tie",
    identical: "Tie · same transcript",
    insufficient: "Insufficient evidence",
    inconsistent: "Inconclusive · Jev disagrees",
  } as const
  for (const [winner, expected] of Object.entries(labels)) {
    assert.equal(
      judgmentLabel({
        source: "jev",
        state: "done",
        winner: winner as keyof typeof labels,
      }),
      expected
    )
  }
})
