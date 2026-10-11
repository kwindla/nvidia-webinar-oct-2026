import { Panel } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { noiseProfiles, useNoise, type NoiseProfile } from "@/stores/noise"

export function NoisePanel() {
  const { profile, volume, status, error, select, setVolume, play, stop } =
    useNoise()
  return (
    <Panel title="background noise" className="shrink-0">
      <div className="space-y-2 p-3 pt-4">
        <div className="flex items-end gap-3">
          <div className="min-w-0 flex-1">
            <label
              htmlFor="noise-profile"
              className="mb-1.5 block text-xs text-muted-foreground"
            >
              Noise profile
            </label>
            <select
              id="noise-profile"
              value={profile}
              onChange={(event) => select(event.target.value as NoiseProfile)}
              className="h-9 w-full border border-border bg-background px-2 text-xs"
            >
              {noiseProfiles.map((name) => (
                <option key={name} value={name}>
                  {name === "all" ? "All airports" : name}
                </option>
              ))}
            </select>
          </div>
          <Button
            variant="outline"
            onClick={() => (status === "stopped" ? void play() : stop())}
          >
            {status === "stopped" ? "Play noise" : "Stop noise"}
          </Button>
        </div>
        <div>
          <label
            htmlFor="noise-volume"
            className="mb-1 block text-xs text-muted-foreground"
          >
            Noise volume · {Math.round(volume * 100)}%
          </label>
          <input
            id="noise-volume"
            aria-label="Noise volume"
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.round(volume * 100)}
            onChange={(event) => setVolume(Number(event.target.value) / 100)}
            className="w-full accent-agent"
          />
        </div>
        <p role="status" className="text-xs text-muted-foreground">
          {status === "playing"
            ? "Playing · loops every 60 seconds"
            : status === "loading"
              ? "Loading noise…"
              : "Stopped · choose a profile, then play"}
        </p>
        {error && (
          <p role="alert" className="text-xs text-inactive">
            {error}
          </p>
        )}
        <p className="text-[11px] text-muted-foreground">
          Uses default speakers. Start quietly. Mic noise suppression may reduce
          the effect.
        </p>
      </div>
    </Panel>
  )
}
