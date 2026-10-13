import { PipecatClient } from "@pipecat-ai/client-js"
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport"
import { create } from "zustand"
import { useComparison } from "@/stores/comparison"
import { useMetrics } from "@/stores/metrics"

export type ModelChoice = "cloud" | "local" | "tuned"
const modelChoices: readonly ModelChoice[] = ["cloud", "local", "tuned"]
let sessionGeneration = 0
// Only the newest /models request may update the store or clear its busy flag.
let catalogRequest = 0

const isModelChoice = (value: unknown): value is ModelChoice =>
  modelChoices.includes(value as ModelChoice)

type ModelNames = Partial<Record<ModelChoice, string>>

function namesFrom(value: unknown): ModelNames {
  if (typeof value !== "object" || value === null) return {}
  return Object.fromEntries(
    Object.entries(value).filter(
      ([key, name]) => isModelChoice(key) && typeof name === "string"
    )
  )
}

function modelFromResponse(response: unknown): {
  model: ModelChoice
  available: ModelChoice[]
  names: ModelNames
} {
  if (
    typeof response === "object" &&
    response !== null &&
    "model" in response &&
    isModelChoice(response.model)
  ) {
    const listed =
      "available" in response && Array.isArray(response.available)
        ? response.available.filter(isModelChoice)
        : []
    return {
      model: response.model,
      available: listed.length ? listed : [response.model],
      names: "names" in response ? namesFrom(response.names) : {},
    }
  }
  throw new Error("The server returned an invalid model selection.")
}

// GET /models lists what the next session can start on, before connecting.
function catalogFromResponse(response: unknown): {
  model: ModelChoice
  available: ModelChoice[]
  names: ModelNames
} {
  if (
    typeof response === "object" &&
    response !== null &&
    "models" in response &&
    Array.isArray(response.models)
  ) {
    const entries = response.models.filter(
      (m): m is { id: ModelChoice; name: string } =>
        typeof m === "object" &&
        m !== null &&
        isModelChoice(m.id) &&
        typeof m.name === "string"
    )
    const available = entries.map((m) => m.id)
    const preferred = "default" in response ? response.default : undefined
    if (available.length)
      return {
        model:
          isModelChoice(preferred) && available.includes(preferred)
            ? preferred
            : available[0],
        available,
        names: Object.fromEntries(entries.map((m) => [m.id, m.name])),
      }
  }
  throw new Error("The server returned an invalid model list.")
}

interface SessionState {
  busy: boolean
  error: string | null
  speaking: boolean
  model: ModelChoice
  availableModels: ModelChoice[]
  modelNames: ModelNames
  modelPicked: boolean
  modelBusy: boolean
  modelError: string | null
  loadModels: () => Promise<void>
  selectModel: (model: ModelChoice) => Promise<void>
  syncModel: () => Promise<void>
  connect: () => Promise<void>
  disconnect: () => Promise<void>
}

// One client per page. Pipecat owns transport, devices and conversation state;
// Zustand owns app actions and presentation state.
export const useSession = create<SessionState>((set, get) => ({
  busy: false,
  error: null,
  speaking: false,
  model: "cloud",
  availableModels: [],
  modelNames: {},
  modelPicked: false,
  modelBusy: true,
  modelError: null,
  loadModels: async () => {
    if (client.state === "ready") return
    const generation = sessionGeneration
    const request = ++catalogRequest
    const current = () =>
      generation === sessionGeneration && request === catalogRequest
    set({ modelBusy: true, modelError: null })
    try {
      const response = await fetch("/models", { cache: "no-store" })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const { model, available, names } = catalogFromResponse(
        await response.json()
      )
      if (current()) {
        const kept = get().model
        set({
          // Keep a choice the user already made if it is still offered.
          model: get().modelPicked && available.includes(kept) ? kept : model,
          availableModels: available,
          modelNames: names,
        })
      }
    } catch {
      if (current())
        set({ modelError: "Could not load the model list. Reload to retry." })
    } finally {
      if (current()) set({ modelBusy: false })
    }
  },
  syncModel: async () => {
    const generation = sessionGeneration
    set({ modelBusy: true })
    try {
      const { model, available, names } = modelFromResponse(
        await client.sendClientRequest("get-model", {}, 10000)
      )
      if (generation === sessionGeneration)
        set({
          model,
          availableModels: available,
          modelNames: names,
          modelError: null,
        })
    } catch {
      if (generation === sessionGeneration)
        set({
          modelError:
            "Could not read the model selection. Choose a model to retry.",
        })
    } finally {
      if (generation === sessionGeneration) set({ modelBusy: false })
    }
  },
  selectModel: async (model) => {
    if (get().modelBusy || !get().availableModels.includes(model)) return
    if (client.state !== "ready") {
      // Before connecting, the choice is sent with the session start.
      if (["disconnected", "initialized", "error"].includes(client.state))
        set({ model, modelPicked: true })
      return
    }
    const generation = sessionGeneration
    set({ modelBusy: true, modelError: null })
    try {
      const { model: selected } = modelFromResponse(
        await client.sendClientRequest("set-model", { model }, 10000)
      )
      if (generation === sessionGeneration) set({ model: selected })
    } catch {
      if (generation === sessionGeneration)
        set({
          modelError: "Could not confirm the switch. Choose a model to retry.",
        })
    } finally {
      if (generation === sessionGeneration) set({ modelBusy: false })
    }
  },
  connect: async () => {
    if (
      get().busy ||
      !["disconnected", "initialized", "error"].includes(client.state)
    )
      return
    sessionGeneration++
    useMetrics.getState().reset()
    useComparison.getState().reset()
    // Send a model only when the user picked one; otherwise the bot's DEFAULT_LLM applies.
    const { model, modelPicked } = get()
    set({ busy: true, error: null, modelBusy: true, modelError: null })
    try {
      if (client.needsInit()) await client.initDevices()
      const response = await client.startBot({
        endpoint: "/start",
        requestData: {
          transport: "webrtc",
          enableDefaultIceServers: true,
          body: modelPicked ? { model } : {},
        },
      })
      await client.connect(response)
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Could not connect. Please try again.",
      })
      await client.disconnect().catch(() => {})
      void get().loadModels()
    } finally {
      set({ busy: false })
    }
  },
  disconnect: async () => {
    if (get().busy) return
    set({ busy: true })
    try {
      await client.disconnect()
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Could not disconnect.",
      })
    } finally {
      set({ busy: false, speaking: false })
    }
  },
}))

export const client = new PipecatClient({
  transport: new SmallWebRTCTransport(),
  enableMic: true,
  enableCam: false,
  callbacks: {
    onServerMessage: (message) => useComparison.getState().ingest(message),
    onMetrics: (metrics) => useMetrics.getState().ingest(metrics),
    onBotStartedSpeaking: () => useSession.setState({ speaking: true }),
    onBotStoppedSpeaking: () => useSession.setState({ speaking: false }),
    onBotReady: () => {
      void useSession.getState().syncModel()
      const generation = sessionGeneration
      void client
        .sendClientRequest("get-comparison", {}, 10000)
        .then((data) => {
          if (generation === sessionGeneration)
            useComparison.getState().ingest(data)
        })
        .catch(() => {})
    },
    onDisconnected: () => {
      useComparison.getState().freeze()
      sessionGeneration++
      useSession.setState({ speaking: false })
      // Refresh what the next session can start on.
      void useSession.getState().loadModels()
    },
    onError: () =>
      useSession.setState({
        error: "The voice session reported an error. Disconnect and try again.",
      }),
  },
})

void useSession.getState().loadModels()

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    void client.disconnect().catch(() => {})
  })
}
