import { Panel } from "@/components/panel"
import {
  RECENT_COUNT,
  WINDOW_SIZE,
  headline,
  latencyCeiling,
  type LatencySeries,
  type TimingKind,
} from "@/lib/latency"
import { useMetrics } from "@/stores/metrics"

const labels: Record<string, string> = {
  "nemotron-asr": "Nemotron ASR",
  "cloud-llm": "GPT-5.6 Terra",
  "local-llm": "gb-phone",
  "tuned-llm": "ATC tutor (tuned)",
  "deepgram-flux-tts": "Deepgram Flux",
}
const descriptions: Record<TimingKind, string> = {
  ttfs: "VAD stop (STT commit) → final transcript. Excludes the VAD silence wait.",
  ttft: "Request → first answer token (Pipecat TTFAT). Includes any preceding reasoning.",
  ttfb: "Request → first output from the service (Pipecat TTFB); for TTS, first audio bytes.",
}

/** The last few samples as numbers: older in blue, the newest in green. */
function Recent({ series }: { series: LatencySeries }) {
  const recent = series.points.slice(-RECENT_COUNT)
  return (
    <p
      className="flex flex-wrap items-baseline gap-x-2 text-[11px] tabular-nums"
      aria-label={`${series.kind.toUpperCase()}, last ${recent.length} samples, newest last: ${recent.map((v) => Math.round(v)).join(", ")} milliseconds`}
    >
      <span className="text-[10px] text-muted-foreground">
        {series.kind.toUpperCase()}
      </span>
      {recent.map((value, index) => (
        <span
          key={series.points.length - recent.length + index}
          className={
            index === recent.length - 1
              ? "font-semibold text-agent"
              : "text-client"
          }
        >
          {Math.round(value)}
        </span>
      ))}
      <span className="text-[10px] text-muted-foreground">ms</span>
    </p>
  )
}

function Sparkline({
  series,
  ceiling,
}: {
  series: LatencySeries
  ceiling: number
}) {
  const coordinates = series.points.map((value, index) => ({
    x:
      2 +
      ((WINDOW_SIZE - series.points.length + index) / (WINDOW_SIZE - 1)) * 116,
    y: 28 - (value / ceiling) * 25,
  }))
  const last = coordinates.at(-1)!
  return (
    <svg
      viewBox="0 0 120 32"
      preserveAspectRatio="none"
      aria-hidden="true"
      className="mt-1 h-8 w-full text-client"
    >
      <title>
        {series.points.map((v) => `${Math.round(v)} ms`).join(" → ")}
      </title>
      <path d="M2 28H118" className="stroke-border" fill="none" />
      <polyline
        points={coordinates.map(({ x, y }) => `${x},${y}`).join(" ")}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx={last.x} cy={last.y} r="2" className="fill-agent" />
    </svg>
  )
}

export function LatencyPanel() {
  const history = useMetrics((s) => s.history)
  const groups = new Map<string, LatencySeries[]>()
  for (const item of Object.values(history)) {
    const key = JSON.stringify([item.processor, item.model])
    groups.set(key, [...(groups.get(key) ?? []), item])
  }
  // One metric per service row; the shared scale covers only what is shown.
  const rows = [...groups.entries()].flatMap(([key, items]) => {
    const shown = headline(items)
    return shown ? [{ key, series: shown }] : []
  })
  const ceiling = latencyCeiling(rows.map((row) => row.series))
  return (
    <Panel title="model latency" className="shrink-0">
      <div className="space-y-3 px-4 pt-5 pb-3">
        <div className="flex flex-wrap justify-between gap-1 text-[10px] text-muted-foreground">
          <span>
            Last {RECENT_COUNT} values · newest in green · line: last{" "}
            {WINDOW_SIZE}
          </span>
          <span>Shared scale: 0–{ceiling} ms</span>
        </div>
        {rows.length === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">
            Waiting for model timing reports…
          </p>
        ) : (
          rows.map(({ key, series }) => (
            <div
              key={key}
              className="grid grid-cols-[100px_minmax(0,1fr)] items-center gap-3 border-t border-border pt-2"
            >
              <div
                className="min-w-0 text-[11px]"
                title={`${series.processor}${series.model ? ` · ${series.model}` : ""}`}
              >
                <p className="break-words">
                  {labels[series.processor] ?? series.processor}
                </p>
                <p className="mt-1 text-[9px] text-muted-foreground">
                  {series.points.length} reports
                </p>
              </div>
              <div className="min-w-0" title={descriptions[series.kind]}>
                <Recent series={series} />
                <Sparkline series={series} ceiling={ceiling} />
              </div>
            </div>
          ))
        )}
        <details className="text-[10px] text-muted-foreground">
          <summary className="cursor-pointer">What the timings mean</summary>
          <dl className="mt-2 space-y-1">
            {Object.entries(descriptions).map(([kind, text]) => (
              <div key={kind}>
                <dt className="inline font-medium">{kind.toUpperCase()}: </dt>
                <dd className="inline">{text}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2">
            Each point is a service report, not a complete conversation turn.
            Only reported metrics appear. TTFB reports can include interrupted
            requests. History resets on a new connection.
          </p>
        </details>
      </div>
    </Panel>
  )
}
