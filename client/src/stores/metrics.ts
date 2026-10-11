import { create } from "zustand"
import { appendLatency, type LatencyHistory } from "@/lib/latency"

export const useMetrics = create<{
  history: LatencyHistory
  ingest: (payload: unknown) => void
  reset: () => void
}>((set) => ({
  history: {},
  ingest: (payload) =>
    set((state) => {
      const history = appendLatency(state.history, payload)
      return history === state.history ? state : { history }
    }),
  reset: () => set({ history: {} }),
}))
