export const WINDOW_SIZE = 30
export type TimingKind = "ttfb" | "ttfs" | "ttft" | "ttfa"
export interface LatencySeries {
  processor: string
  model?: string
  kind: TimingKind
  source: string
  points: number[] // milliseconds, oldest first; one point per service report
}
export type LatencyHistory = Record<string, LatencySeries>

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Consume measured headline values only; nested TTFB breakdowns are duplicates. */
export function appendLatency(
  history: LatencyHistory,
  payload: unknown
): LatencyHistory {
  if (!record(payload)) return history
  let next = history
  for (const [source, field] of Object.entries({
    ttfb: "value",
    ttfa: "ttfa",
    ttfat: "ttfat",
    ttfs: "value",
    ttft: "value",
  })) {
    const entries = payload[source]
    if (!Array.isArray(entries)) continue
    for (const entry of entries) {
      if (
        !record(entry) ||
        typeof entry.processor !== "string" ||
        !entry.processor
      )
        continue
      const seconds = entry[field]
      if (
        typeof seconds !== "number" ||
        !Number.isFinite(seconds) ||
        seconds < 0
      )
        continue
      const model = typeof entry.model === "string" ? entry.model : undefined
      const kind: TimingKind =
        source === "ttfat"
          ? "ttft"
          : source === "ttfb" && entry.processor === "nemotron-asr"
            ? "ttfs"
            : (source as TimingKind)
      const key = JSON.stringify([entry.processor, model ?? "", kind])
      if (next === history) next = { ...history }
      next[key] = {
        processor: entry.processor,
        model,
        kind,
        source,
        points: [...(next[key]?.points ?? []), seconds * 1000].slice(
          -WINDOW_SIZE
        ),
      }
    }
  }
  return next
}

/** One zero-based vertical scale across every chart, rounded to 1/2/5 steps. */
export function latencyCeiling(series: LatencySeries[]): number {
  const max = Math.max(500, ...series.flatMap((s) => s.points))
  const power = 10 ** Math.floor(Math.log10(max))
  return [1, 2, 5, 10].map((n) => n * power).find((n) => n >= max)!
}
