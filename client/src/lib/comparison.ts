export type Source = "fine-tuned" | "base"
export interface Transcript {
  source: Source
  model: string
  final: string
  interim: string
  state: "pending" | "final" | "unavailable"
}
export interface Judgment {
  source: "jev" | "local"
  state: "waiting" | "pending" | "done" | "unavailable" | "skipped"
  reason?: string
  winner?:
    | "fine-tuned"
    | "base"
    | "tie"
    | "insufficient"
    | "inconsistent"
    | "identical"
  mode?: "reference" | "context" | "text_agreement"
  confidence?: number
  model?: string
  context_turns?: number
  elapsed_ms?: number
}
export interface ComparisonTurn {
  id: number
  closed: boolean
  reference: string
  sources: Record<Source, Transcript>
  judge: Judgment
  wer?: Partial<Record<Source, number | null>> | null
}
export interface ComparisonSnapshot {
  type: "asr-comparison"
  version: 1
  session_id: string
  revision: number
  status: Record<Source, string>
  errors: Record<Source, string | null>
  models: Record<Source, string>
  reference: string
  context_judging: boolean
  turns: ComparisonTurn[]
}
function object(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v)
}
function text(v: unknown, limit: number): v is string {
  return typeof v === "string" && v.length <= limit
}
const sources: Source[] = ["fine-tuned", "base"]
export function parseComparison(v: unknown): ComparisonSnapshot | null {
  if (
    !object(v) ||
    v.type !== "asr-comparison" ||
    v.version !== 1 ||
    !text(v.session_id, 100) ||
    !Number.isSafeInteger(v.revision) ||
    Number(v.revision) < 0 ||
    !text(v.reference, 500) ||
    typeof v.context_judging !== "boolean" ||
    !object(v.status) ||
    !object(v.errors) ||
    !object(v.models) ||
    !sources.every(
      (s) =>
        text((v.status as Record<string, unknown>)[s], 30) &&
        text((v.models as Record<string, unknown>)[s], 200) &&
        ((v.errors as Record<string, unknown>)[s] === null ||
          text((v.errors as Record<string, unknown>)[s], 100))
    ) ||
    !Array.isArray(v.turns) ||
    v.turns.length > 20
  )
    return null
  const ids = new Set<number>()
  for (const t of v.turns) {
    if (
      !object(t) ||
      !Number.isSafeInteger(t.id) ||
      Number(t.id) < 1 ||
      ids.has(Number(t.id)) ||
      typeof t.closed !== "boolean" ||
      !text(t.reference, 500) ||
      !object(t.sources) ||
      !object(t.judge)
    )
      return null
    ids.add(Number(t.id))
    for (const s of sources) {
      const transcript = t.sources[s]
      if (
        !object(transcript) ||
        transcript.source !== s ||
        !text(transcript.model, 200) ||
        !text(transcript.final, 2000) ||
        !text(transcript.interim, 2000) ||
        !["pending", "final", "unavailable"].includes(String(transcript.state))
      )
        return null
    }
    const j = t.judge
    if (j.state === "done" && (j.winner === undefined || j.mode === undefined))
      return null
    if (
      !["jev", "local"].includes(String(j.source)) ||
      !["waiting", "pending", "done", "unavailable", "skipped"].includes(
        String(j.state)
      ) ||
      (j.reason !== undefined && !text(j.reason, 100)) ||
      (j.winner !== undefined &&
        ![
          "fine-tuned",
          "base",
          "tie",
          "insufficient",
          "inconsistent",
          "identical",
        ].includes(String(j.winner))) ||
      (j.mode !== undefined &&
        !["reference", "context", "text_agreement"].includes(String(j.mode))) ||
      (j.context_turns !== undefined &&
        (!Number.isInteger(j.context_turns) ||
          Number(j.context_turns) < 0 ||
          Number(j.context_turns) > 10)) ||
      (j.confidence !== undefined &&
        (typeof j.confidence !== "number" ||
          !Number.isFinite(j.confidence) ||
          j.confidence < 0 ||
          j.confidence > 1)) ||
      (j.model !== undefined && !text(j.model, 200)) ||
      (j.elapsed_ms !== undefined &&
        (typeof j.elapsed_ms !== "number" ||
          !Number.isFinite(j.elapsed_ms) ||
          j.elapsed_ms < 0))
    )
      return null
    if (
      t.wer != null &&
      (!object(t.wer) ||
        !Object.values(t.wer).every(
          (w) =>
            w === null ||
            (typeof w === "number" && Number.isFinite(w) && w >= 0)
        ))
    )
      return null
  }
  return v as unknown as ComparisonSnapshot
}

export function acceptComparison(
  previous: ComparisonSnapshot | null,
  payload: unknown
) {
  const next = parseComparison(payload)
  if (
    !next ||
    (previous &&
      (next.session_id !== previous.session_id ||
        next.revision <= previous.revision))
  )
    return previous
  return next
}

export function judgmentLabel(j: Judgment): string {
  if (j.state === "waiting") return "Awaiting final transcripts"
  if (j.state === "pending") return "Jev judging…"
  if (j.state === "unavailable") return "Judgment unavailable"
  if (j.state === "skipped") return "Not judged"
  switch (j.winner) {
    case "fine-tuned":
      return "Fine-tuned better"
    case "base":
      return "Base better"
    case "tie":
      return "Tie"
    case "identical":
      return "Tie · same transcript"
    case "inconsistent":
      return "Inconclusive · Jev disagrees"
    default:
      return "Insufficient evidence"
  }
}
