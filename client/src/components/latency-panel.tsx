import { Panel } from "@/components/panel"
import { Metric } from "@/components/pipecat/metric"
import {
  WINDOW_SIZE,
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
  ttfs: "Speech end → final transcript (Pipecat STT TTFB). Includes VAD stop delay.",
  ttft: "Request → first answer token (Pipecat TTFAT). Includes any preceding reasoning.",
  ttfb: "Request → first output from the service (Pipecat TTFB).",
  ttfa: "TTS request → first audible sample (Pipecat TTFA). Includes leading silence; excludes browser playback delay.",
}
const colors: Record<TimingKind, string> = {
  ttfs: "text-client",
  ttft: "text-agent",
  ttfb: "text-tool",
  ttfa: "text-active",
}

function Sparkline({
  series,
  ceiling,
}: {
  series: LatencySeries
  ceiling: number
}) {
  const latest = series.points.at(-1)!
  const coordinates = series.points.map((value, index) => ({
    x:
      2 +
      ((WINDOW_SIZE - series.points.length + index) / (WINDOW_SIZE - 1)) * 116,
    y: 28 - (value / ceiling) * 25,
  }))
  const last = coordinates.at(-1)!
  return (
    <div
      title={descriptions[series.kind]}
      className={`min-w-0 ${colors[series.kind]}`}
    >
      <Metric
        label={series.kind.toUpperCase()}
        value={`${Math.round(latest)} ms`}
        className="gap-1 text-[10px]"
        valueClassName="text-current"
      />
      <svg
        viewBox="0 0 120 32"
        preserveAspectRatio="none"
        role="img"
        className="mt-1 h-8 w-full"
        aria-label={`${series.kind.toUpperCase()}, latest ${Math.round(latest)} milliseconds, ${series.points.length} samples, range zero to ${ceiling} milliseconds`}
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
        <circle cx={last.x} cy={last.y} r="2" fill="currentColor" />
      </svg>
    </div>
  )
}

export function LatencyPanel() {
  const history = useMetrics((s) => s.history)
  const series = Object.values(history)
  const ceiling = latencyCeiling(series)
  const groups = new Map<string, LatencySeries[]>()
  for (const item of series) {
    const key = JSON.stringify([item.processor, item.model])
    groups.set(key, [...(groups.get(key) ?? []), item])
  }
  return (
    <Panel title="model latency" className="shrink-0">
      <div className="space-y-3 px-4 pt-5 pb-3">
        <div className="flex flex-wrap justify-between gap-1 text-[10px] text-muted-foreground">
          <span>Last {WINDOW_SIZE} reports · oldest → newest</span>
          <span>Shared scale: 0–{ceiling} ms</span>
        </div>
        {groups.size === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">
            Waiting for model timing reports…
          </p>
        ) : (
          [...groups.entries()].map(([key, items]) => (
            <div
              key={key}
              className="grid grid-cols-[100px_minmax(0,1fr)] items-center gap-3 border-t border-border pt-2"
            >
              <div
                className="min-w-0 text-[11px]"
                title={`${items[0].processor}${items[0].model ? ` · ${items[0].model}` : ""}`}
              >
                <p className="break-words">
                  {labels[items[0].processor] ?? items[0].processor}
                </p>
                <p className="mt-1 text-[9px] text-muted-foreground">
                  {Math.max(...items.map((s) => s.points.length))} reports
                </p>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                {items.map((item) => (
                  <Sparkline key={item.kind} series={item} ceiling={ceiling} />
                ))}
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
