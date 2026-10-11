import { usePipecatClientTransportState } from "@pipecat-ai/client-react"
import { Button } from "@/components/ui/button"
import { useSession, type ModelChoice } from "@/stores/session"

const choices: { value: ModelChoice; label: string; name: string }[] = [
  { value: "cloud", label: "Cloud", name: "GPT-5.6 Terra" },
  { value: "local", label: "Local", name: "gb-phone" },
  { value: "tuned", label: "Tuned", name: "ATC tutor" },
]

export function ModelSelector() {
  const state = usePipecatClientTransportState()
  const model = useSession((s) => s.model)
  const available = useSession((s) => s.availableModels)
  const shown = choices.filter((choice) => available.includes(choice.value))
  const busy = useSession((s) => s.modelBusy)
  const error = useSession((s) => s.modelError)
  const select = useSession((s) => s.selectModel)
  const ready = state === "ready"
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground" id="model-label">
        Response model
      </p>
      <div
        role="group"
        aria-labelledby="model-label"
        className={`grid gap-2 ${shown.length > 2 ? "grid-cols-3" : "grid-cols-2"}`}
      >
        {shown.map((choice) => (
          <Button
            key={choice.value}
            variant="outline"
            aria-pressed={model === choice.value}
            disabled={!ready || busy}
            onClick={() => void select(choice.value)}
            className={`h-auto min-w-0 flex-col gap-0.5 py-1.5 ${model === choice.value ? "border-active text-active" : ""}`}
          >
            <span>{choice.label}</span>
            <span className="text-xs">{choice.name}</span>
          </Button>
        ))}
      </div>
      <p role="status" className="text-xs text-muted-foreground">
        {!ready
          ? "Connect to switch models."
          : busy
            ? "Confirming selection…"
            : "Applies to the next response. Conversation is preserved."}
      </p>
      {error && (
        <p role="alert" className="text-xs text-inactive">
          {error}
        </p>
      )}
    </div>
  )
}
