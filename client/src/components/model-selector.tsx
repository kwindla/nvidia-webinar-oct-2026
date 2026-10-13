import { usePipecatClientTransportState } from "@pipecat-ai/client-react"
import { Button } from "@/components/ui/button"
import { useSession, type ModelChoice } from "@/stores/session"

const labels: Record<ModelChoice, string> = {
  cloud: "Cloud",
  local: "Local",
  tuned: "Tuned",
}

export function ModelSelector() {
  const state = usePipecatClientTransportState()
  const model = useSession((s) => s.model)
  const available = useSession((s) => s.availableModels)
  const names = useSession((s) => s.modelNames)
  const busy = useSession((s) => s.modelBusy)
  const sessionBusy = useSession((s) => s.busy)
  const error = useSession((s) => s.modelError)
  const select = useSession((s) => s.selectModel)
  const ready = state === "ready"
  // Before connecting, a choice sets the model the session starts on.
  const idle = ["disconnected", "initialized", "error"].includes(state)
  const enabled = (ready || (idle && !sessionBusy)) && !busy
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground" id="model-label">
        Response model
      </p>
      <div
        role="group"
        aria-labelledby="model-label"
        className={`grid gap-2 ${available.length > 2 ? "grid-cols-3" : "grid-cols-2"}`}
      >
        {available.map((choice) => (
          <Button
            key={choice}
            variant="outline"
            aria-pressed={model === choice}
            disabled={!enabled}
            onClick={() => void select(choice)}
            className={`h-auto min-w-0 flex-col gap-0.5 py-1.5 ${model === choice ? "border-active text-active" : ""}`}
          >
            <span>{labels[choice]}</span>
            {names[choice] && <span className="text-xs">{names[choice]}</span>}
          </Button>
        ))}
      </div>
      <p role="status" className="text-xs text-muted-foreground">
        {busy
          ? ready
            ? "Confirming selection…"
            : "Loading models…"
          : ready
            ? "Applies to the next response. Conversation is preserved."
            : idle
              ? "The session starts with this model."
              : "Connecting…"}
      </p>
      {error && (
        <p role="alert" className="text-xs text-inactive">
          {error}
        </p>
      )}
    </div>
  )
}
