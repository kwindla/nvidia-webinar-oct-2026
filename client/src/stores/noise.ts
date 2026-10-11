import { create } from "zustand"

export const noiseProfiles = [
  "all",
  "LKPR",
  "LKTB",
  "LSGS",
  "LSZB",
  "LSZH",
  "LZIB",
  "YSSY",
] as const
export type NoiseProfile = (typeof noiseProfiles)[number]
let audio: HTMLAudioElement | undefined
let generation = 0

interface NoiseState {
  profile: NoiseProfile
  volume: number
  status: "stopped" | "loading" | "playing"
  error: string | null
  select: (profile: NoiseProfile) => void
  setVolume: (volume: number) => void
  play: () => Promise<void>
  stop: () => void
}

export const useNoise = create<NoiseState>((set, get) => ({
  profile: "all",
  volume: 0.2,
  status: "stopped",
  error: null,
  select: (profile) => {
    get().stop()
    set({ profile })
  },
  setVolume: (volume) => {
    const next = Math.max(0, Math.min(1, volume))
    if (audio) audio.volume = next
    set({ volume: next })
  },
  play: async () => {
    get().stop()
    const current = ++generation
    const player = new Audio(`/noise/noise-${get().profile}.wav`)
    audio = player
    player.loop = true
    player.volume = get().volume
    player.addEventListener("error", () => {
      if (current !== generation) return
      get().stop()
      set({
        error:
          "Noise audio could not load. Try again or choose another profile.",
      })
    })
    set({ status: "loading", error: null })
    try {
      await player.play()
      if (current === generation) set({ status: "playing" })
      else player.pause()
    } catch {
      if (current !== generation) return
      get().stop()
      set({
        error:
          "Playback could not start. Try again and check browser audio permissions.",
      })
    }
  },
  stop: () => {
    generation++
    if (audio) {
      audio.pause()
      audio.removeAttribute("src")
      audio.load()
      audio = undefined
    }
    set({ status: "stopped", error: null })
  },
}))

// One page-level player, independent of bot sessions. Never autoplay on navigation.
const stopOnLeave = () => useNoise.getState().stop()
window.addEventListener("pagehide", stopOnLeave)
if (import.meta.hot)
  import.meta.hot.dispose(() => {
    stopOnLeave()
    window.removeEventListener("pagehide", stopOnLeave)
  })
