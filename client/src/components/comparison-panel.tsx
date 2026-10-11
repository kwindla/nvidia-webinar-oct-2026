import { useState } from "react"
import { usePipecatClientTransportState } from "@pipecat-ai/client-react"
import { Panel } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { useComparison } from "@/stores/comparison"
import { client } from "@/stores/session"
import { judgmentLabel, type Judgment, type Source } from "@/lib/comparison"

const sources: Source[] = ["fine-tuned", "base"]
const labels = { "fine-tuned": "Fine-tuned EA", base: "Base Nemotron 3.5" }
function verdict(j: Judgment) {
  if (j.state === "waiting")
    return "Waiting for turn end and both final results…"
  if (j.state === "pending") return "Jev is comparing both candidate orders…"
  if (j.state === "skipped")
    return j.reason === "no_audio_segments"
      ? "No audio segments to compare"
      : "Context judging disabled"
  if (j.state === "unavailable")
    return `Comparison unavailable · ${j.reason?.replaceAll("_", " ") ?? "unknown error"}`
  if (j.winner === "identical")
    return "Same normalized transcript · local text agreement"
  const winner =
    j.winner === "fine-tuned" || j.winner === "base"
      ? labels[j.winner]
      : j.winner
  return `${j.mode === "reference" ? "Reference-based preference" : "Context-based preference"}: ${winner}${j.confidence === undefined ? "" : ` · min. choice confidence ${Math.round(j.confidence * 100)}%`}`
}
export function ComparisonPanel() {
  const snapshot = useComparison((s) => s.snapshot)
  const rejected = useComparison((s) => s.rejected)
  const ready = usePipecatClientTransportState() === "ready"
  const [reference, setReference] = useState("")
  const [contextJudging, setContextJudging] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  async function apply() {
    setBusy(true)
    setError(null)
    try {
      const session = useComparison.getState().snapshot?.session_id
      const data = await client.sendClientRequest(
        "set-comparison",
        {
          reference,
          context_judging: contextJudging ?? snapshot?.context_judging ?? true,
        },
        10000
      )
      if (
        session !== useComparison.getState().snapshot?.session_id ||
        !useComparison.getState().accepting
      )
        return
      useComparison.getState().ingest(data)
      setReference("")
      setContextJudging(null)
    } catch {
      setError("Could not confirm settings. Try again.")
    } finally {
      setBusy(false)
    }
  }
  return (
    <Panel title="ASR comparison" className="flex min-h-0 flex-col">
      <div className="min-h-0 space-y-2 overflow-y-auto p-3 pt-4">
        {rejected >= 3 && (
          <p role="alert" className="text-xs text-inactive">
            Comparison data could not be read. Reconnect to resynchronize.
          </p>
        )}
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {sources.map((s) => (
            <div
              key={s}
              className="min-w-0 text-xs"
              title={snapshot?.models[s]}
            >
              <span className="text-agent">{labels[s]}</span> ·{" "}
              {snapshot?.status[s] ?? "Connect to compare"}
              {snapshot?.errors[s] && (
                <p className="break-words text-inactive">
                  {snapshot.errors[s]?.replaceAll("_", " ")}
                </p>
              )}
            </div>
          ))}
        </div>
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">
            Reference and judging settings
          </summary>
          <div className="mt-3 space-y-2">
            <label htmlFor="comparison-reference" className="block">
              Reference for the next microphone turn
            </label>
            <textarea
              id="comparison-reference"
              maxLength={500}
              rows={2}
              value={reference}
              onChange={(e) =>
                setReference(e.target.value.replace(/[\r\n]/g, " "))
              }
              placeholder="Optional: enter the phrase you will say"
              className="w-full resize-y border border-border bg-background p-2"
            />
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={contextJudging ?? snapshot?.context_judging ?? true}
                onChange={(e) => setContextJudging(e.target.checked)}
              />
              Judge each turn using conversation context
            </label>
            <Button
              variant="outline"
              disabled={!ready || !snapshot || busy}
              onClick={() => void apply()}
            >
              Apply for next turn
            </Button>
            <p role="status" className="break-words text-muted-foreground">
              {snapshot?.reference
                ? `Armed once: ${snapshot.reference}`
                : "No reference armed"}{" "}
              · Context judging {snapshot?.context_judging ? "on" : "off"}
            </p>
            <p className="text-muted-foreground">
              Speak the reference as one turn. A pause that ends the turn can
              split the phrase. Jev receives both transcripts, your optional
              reference, and up to 10 prior user/assistant turns. Context-based
              preferences infer the intended words; Jev cannot hear the audio.
            </p>
            {error && (
              <p role="alert" className="text-inactive">
                {error}
              </p>
            )}
          </div>
        </details>
        <div className="space-y-3">
          {!snapshot?.turns.length && (
            <p className="text-xs text-muted-foreground">
              Speak to see both transcripts here.
            </p>
          )}
          {[...(snapshot?.turns ?? [])].reverse().map((t) => (
            <article
              key={t.id}
              className="space-y-2 border-t border-border pt-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-[11px] text-muted-foreground">
                  Turn {t.id} · {t.closed ? "Ended" : "Listening"}
                </p>
                <span
                  role="status"
                  className={`border px-2 py-1 text-xs font-medium ${t.judge.winner === "fine-tuned" ? "border-agent text-agent" : t.judge.winner === "base" ? "border-client text-client" : "border-border text-muted-foreground"}`}
                >
                  {judgmentLabel(t.judge)}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {sources.map((s) => (
                  <div
                    key={s}
                    className="min-w-0 border border-border p-2 text-xs"
                  >
                    <p className="mb-2 text-[10px] text-agent">
                      {labels[s]} · {t.sources[s].state}
                    </p>
                    <p className="break-words">
                      <span>{t.sources[s].final}</span>
                      {t.sources[s].final && t.sources[s].interim ? " " : ""}
                      <em>{t.sources[s].interim}</em>
                      {!t.sources[s].final &&
                        !t.sources[s].interim &&
                        (t.sources[s].state === "final"
                          ? "(No speech recognized)"
                          : "…")}
                    </p>
                  </div>
                ))}
              </div>
              {t.reference && (
                <p className="text-[11px] break-words text-muted-foreground">
                  Reference: {t.reference}
                </p>
              )}
              <p className="text-xs" title={t.judge.model}>
                {verdict(t.judge)}
              </p>
              {t.judge.model && (
                <p className="text-[10px] text-muted-foreground">
                  {t.judge.model} · {t.judge.elapsed_ms} ms ·{" "}
                  {t.judge.context_turns ?? 0} prior turns
                </p>
              )}
              {t.wer && (
                <p className="text-[10px] text-muted-foreground">
                  Raw WER:{" "}
                  {sources
                    .map(
                      (s) =>
                        `${labels[s]} ${t.wer?.[s] == null ? "n/a" : `${Math.round(t.wer[s]! * 100)}%`}`
                    )
                    .join(" · ")}
                  . Jev permits equivalent number formatting; raw WER counts it
                  as word errors; this assumes you spoke the reference.
                </p>
              )}
            </article>
          ))}
        </div>
      </div>
    </Panel>
  )
}
