import { create } from "zustand"
import {
  parseComparison,
  acceptComparison,
  type ComparisonSnapshot,
} from "@/lib/comparison"

export const useComparison = create<{
  snapshot: ComparisonSnapshot | null
  accepting: boolean
  rejected: number
  ingest: (data: unknown) => void
  reset: () => void
  freeze: () => void
}>((set) => ({
  snapshot: null,
  accepting: false,
  rejected: 0,
  ingest: (data) =>
    set((s) => {
      if (
        !s.accepting ||
        !data ||
        typeof data !== "object" ||
        !("type" in data) ||
        data.type !== "asr-comparison"
      )
        return s
      if (!parseComparison(data)) return { rejected: s.rejected + 1 }
      return { snapshot: acceptComparison(s.snapshot, data), rejected: 0 }
    }),
  reset: () => set({ snapshot: null, accepting: true, rejected: 0 }),
  freeze: () => set({ accepting: false }),
}))
