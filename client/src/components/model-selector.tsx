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
  const catalogBusy = useSession((s) => s.catalogBusy)
  const catalogError = useSession((s) => s.catalogError)
  const fallback = useSession((s) => s.fallbackNotice)
  const loadModels = useSession((s) => s.loadModels)
  const syncModel = useSession((s) => s.syncModel)
  const sessionBusy = useSession((s) => s.busy)
  const error = useSession((s) => s.modelError)
  const select = useSession((s) => s.selectModel)
  const ready = state === "ready"
  // Before connecting, a choice sets the model the session starts on.
  const idle = ["disconnected", "initialized", "error"].includes(state)
  const enabled = (ready || (idle && !sessionBusy)) && !busy && !catalogBusy
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
        {catalogBusy
          ? "Loading models…"
          : busy
            ? ready
              ? "Confirming selection…"
              : "Loading models…"
            : ready
              ? "Applies to the next response. Conversation is preserved."
              : idle
                ? "Starts with this model; any fallback is shown after connecting."
                : "Connecting…"}
      </p>
      {fallback && (
        <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">
          {fallback}
        </p>
      )}
      {idle && catalogError && (
        <div>
          <p role="alert" className="text-xs text-inactive">
            {catalogError}
          </p>
          <Button
            variant="outline"
            disabled={catalogBusy || sessionBusy}
            onClick={() => void loadModels()}
          >
            Retry models
          </Button>
        </div>
      )}
      {ready && error && (
        <Button
          variant="outline"
          disabled={busy}
          onClick={() => void syncModel()}
        >
          Refresh model selection
        </Button>
      )}
      {error && (
        <p role="alert" className="text-xs text-inactive">
          {error}
        </p>
      )}
    </div>
  )
}
