import { PipecatClient } from "@pipecat-ai/client-js"
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport"
import { create } from "zustand"
import { useComparison } from "@/stores/comparison"
import { useMetrics } from "@/stores/metrics"

export type ModelChoice = "cloud" | "local" | "tuned"
const modelChoices: readonly ModelChoice[] = ["cloud", "local", "tuned"]
let sessionGeneration = 0

const isModelChoice = (value: unknown): value is ModelChoice =>
  modelChoices.includes(value as ModelChoice)

function modelFromResponse(response: unknown): {
  model: ModelChoice
  available: ModelChoice[]
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
      available: listed.length ? listed : ["cloud", "local"],
    }
  }
  throw new Error("The server returned an invalid model selection.")
}

interface SessionState {
  busy: boolean
  error: string | null
  speaking: boolean
  model: ModelChoice
  availableModels: ModelChoice[]
  modelBusy: boolean
  modelError: string | null
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
  availableModels: ["cloud", "local"],
  modelBusy: true,
  modelError: null,
  syncModel: async () => {
    const generation = sessionGeneration
    set({ modelBusy: true })
    try {
      const { model, available } = modelFromResponse(
        await client.sendClientRequest("get-model", {}, 10000)
      )
      if (generation === sessionGeneration)
        set({ model, availableModels: available, modelError: null })
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
    if (client.state !== "ready" || get().modelBusy) return
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
    set({
      busy: true,
      error: null,
      model: "cloud",
      modelBusy: true,
      modelError: null,
    })
    try {
      if (client.needsInit()) await client.initDevices()
      const response = await client.startBot({
        endpoint: "/start",
        requestData: { transport: "webrtc", enableDefaultIceServers: true },
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
      useSession.setState({ speaking: false, modelBusy: true })
    },
    onError: () =>
      useSession.setState({
        error: "The voice session reported an error. Disconnect and try again.",
      }),
  },
})

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    void client.disconnect().catch(() => {})
  })
}
