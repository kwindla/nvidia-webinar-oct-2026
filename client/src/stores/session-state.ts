import type { PipecatClient } from "@pipecat-ai/client-js"
import type { StateCreator } from "zustand"

type SessionClient = Pick<
  PipecatClient,
  | "state"
  | "sendClientRequest"
  | "needsInit"
  | "initDevices"
  | "startBot"
  | "connect"
  | "disconnect"
>

function fallbackMessage(value: unknown): string | null {
  if (!value || typeof value !== "object") return null
  if (!(
    "requested" in value &&
    typeof value.requested === "string" &&
    "actual" in value &&
    typeof value.actual === "string" &&
    "reason" in value &&
    typeof value.reason === "string"
  ))
    return null
  return `Requested ${value.requested}; using ${value.actual}. ${value.reason}`
}

export type ModelChoice = "cloud" | "local" | "tuned"
const modelChoices: readonly ModelChoice[] = ["cloud", "local", "tuned"]

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
  fallbackNotice: string | null
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
      fallbackNotice: fallbackMessage(
        "fallback" in response ? response.fallback : null
      ),
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

export interface SessionState {
  generation: number
  catalogBusy: boolean
  catalogError: string | null
  fallbackNotice: string | null
  sessionEnded: () => void
  dispose: () => void
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

export function createSessionState({
  getClient,
  onNewSession,
  fetchCatalog = fetch,
  catalogTimeoutMs = 5000,
}: {
  getClient: () => SessionClient
  onNewSession: () => void
  fetchCatalog?: typeof fetch
  catalogTimeoutMs?: number
}): StateCreator<SessionState> {
  let disposed = false
  let sessionGeneration = 0
  let catalogRequest = 0
  let catalogAbort: AbortController | undefined
  const cancelCatalog = () => {
    catalogRequest++
    catalogAbort?.abort()
    catalogAbort = undefined
  }
  return (set, get) => ({
    generation: 0,
    catalogBusy: false,
    catalogError: null,
    fallbackNotice: null,
    dispose: () => {
      disposed = true
      sessionGeneration++
      cancelCatalog()
    },
    sessionEnded: () => {
      if (disposed) return
      sessionGeneration++
      cancelCatalog()
      set({
        generation: sessionGeneration,
        speaking: false,
        modelBusy: false,
        fallbackNotice: null,
      })
      void get().loadModels()
    },
    busy: false,
    error: null,
    speaking: false,
    model: "cloud",
    availableModels: [],
    modelNames: {},
    modelPicked: false,
    modelBusy: false,
    modelError: null,
    loadModels: async () => {
      if (disposed) return
      if (
        !["disconnected", "initialized", "error", "disconnecting"].includes(
          getClient().state
        ) ||
        (get().busy && getClient().state !== "disconnecting")
      )
        return
      cancelCatalog()
      const generation = sessionGeneration
      const request = catalogRequest
      const controller = new AbortController()
      catalogAbort = controller
      const current = () =>
        generation === sessionGeneration && request === catalogRequest
      const timer = setTimeout(() => controller.abort(), catalogTimeoutMs)
      set({ catalogBusy: true, catalogError: null })
      try {
        const response = await fetchCatalog("/models", {
          cache: "no-store",
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const { model, available, names } = catalogFromResponse(
          await response.json()
        )
        if (current()) {
          // Keep an explicit pick even if it disappeared: the server will explain any fallback.
          set({
            model: get().modelPicked ? get().model : model,
            availableModels: available,
            modelNames: names,
          })
        }
      } catch {
        if (current())
          set({
            catalogError:
              "Could not load the model list. Retry to refresh available models.",
          })
      } finally {
        clearTimeout(timer)
        if (current()) {
          catalogAbort = undefined
          set({ catalogBusy: false })
        }
      }
    },
    syncModel: async () => {
      const generation = sessionGeneration
      set({ modelBusy: true })
      try {
        const { model, available, names, fallbackNotice } = modelFromResponse(
          await getClient().sendClientRequest("get-model", {}, 10000)
        )
        if (generation === sessionGeneration)
          set({
            model,
            fallbackNotice,
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
      if (
        get().busy ||
        get().modelBusy ||
        get().catalogBusy ||
        !get().availableModels.includes(model)
      )
        return
      if (getClient().state !== "ready") {
        // Before connecting, the choice is sent with the session start.
        if (
          ["disconnected", "initialized", "error"].includes(getClient().state)
        )
          set({
            model,
            modelPicked: true,
            modelError: null,
            fallbackNotice: null,
          })
        return
      }
      const generation = sessionGeneration
      set({ modelBusy: true, modelError: null })
      try {
        const { model: selected } = modelFromResponse(
          await getClient().sendClientRequest("set-model", { model }, 10000)
        )
        if (generation === sessionGeneration)
          set({ model: selected, modelPicked: true, fallbackNotice: null })
      } catch {
        if (generation === sessionGeneration)
          set({
            modelError:
              "Could not confirm the switch. Choose a model to retry.",
          })
      } finally {
        if (generation === sessionGeneration) set({ modelBusy: false })
      }
    },
    connect: async () => {
      if (
        get().busy ||
        !["disconnected", "initialized", "error"].includes(getClient().state)
      )
        return
      sessionGeneration++
      cancelCatalog()
      onNewSession()
      // Send a model only when the user picked one; otherwise the bot's DEFAULT_LLM applies.
      const { model, modelPicked } = get()
      set({
        generation: sessionGeneration,
        busy: true,
        error: null,
        modelBusy: true,
        modelError: null,
        catalogBusy: false,
        fallbackNotice: null,
      })
      try {
        if (getClient().needsInit()) await getClient().initDevices()
        const response = await getClient().startBot({
          endpoint: "/start",
          requestData: {
            transport: "webrtc",
            enableDefaultIceServers: true,
            body: modelPicked ? { model } : {},
          },
        })
        await getClient().connect(response)
      } catch (error) {
        set({
          error:
            error instanceof Error
              ? error.message
              : "Could not connect. Please try again.",
        })
        await getClient()
          .disconnect()
          .catch(() => {})
      } finally {
        set({ busy: false })
        if (
          ["disconnected", "initialized", "error"].includes(getClient().state)
        ) {
          set({ modelBusy: false })
          void get().loadModels()
        }
      }
    },
    disconnect: async () => {
      if (get().busy) return
      set({ busy: true })
      try {
        await getClient().disconnect()
      } catch (error) {
        set({
          error:
            error instanceof Error ? error.message : "Could not disconnect.",
        })
      } finally {
        set({ busy: false, speaking: false })
      }
    },
  })
}
