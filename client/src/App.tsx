import { Tabs } from "@base-ui/react/tabs"
import {
  PipecatClientProvider,
  useMediaState,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react"
import pipecatLogo from "@/assets/pipecat-logo.svg"
import { Panel } from "@/components/panel"
import { ModelSelector } from "@/components/model-selector"
import { ComparisonPanel } from "@/components/comparison-panel"
import { NoisePanel } from "@/components/noise-panel"
import { LatencyPanel } from "@/components/latency-panel"
import { ConnectButton } from "@/components/pipecat/connect-button"
import { DeviceSelect } from "@/components/pipecat/device-select"
import { AudioVisualizerBar } from "@/components/pipecat/audio-visualizer-bar"
import { BotAudioOutput } from "@/components/pipecat/bot-audio"
import { Conversation } from "@/components/pipecat/conversation"
import { TextInput } from "@/components/pipecat/text-input"
import { UserAudioControl } from "@/components/pipecat/user-audio-control"
import { client, useSession } from "@/stores/session"

function Session() {
  const state = usePipecatClientTransportState()
  const busy = useSession((s) => s.busy)
  const error = useSession((s) => s.error)
  const speaking = useSession((s) => s.speaking)
  const connect = useSession((s) => s.connect)
  const disconnect = useSession((s) => s.disconnect)
  const { mic } = useMediaState()
  const micUnavailable = mic.state === "error"
  const ready = state === "ready"
  const status = ready
    ? speaking
      ? "Speaking"
      : micUnavailable
        ? "Text only"
        : "Listening"
    : busy
      ? "Connecting…"
      : "Ready to connect"
  return (
    <div className="mx-auto flex min-h-svh max-w-[1440px] flex-col gap-4 px-4 py-4 text-sm md:h-svh md:px-6 md:py-4">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="mb-1 text-xs tracking-[0.25em] text-agent uppercase">
            NVIDIA · ASR demo
          </p>
          <h1 className="text-xl font-medium tracking-tight sm:text-2xl">
            Nemotron <span className="text-muted-foreground">/ Voice Lab</span>
          </h1>
        </div>
        <img
          src={pipecatLogo}
          alt="Powered by Pipecat"
          className="h-6 w-auto"
        />
      </header>
      <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 md:grid-cols-[minmax(280px,0.75fr)_minmax(0,1.4fr)]">
        <Tabs.Root
          defaultValue="voice"
          className="flex min-h-0 min-w-0 flex-col gap-4"
        >
          <Tabs.List
            aria-label="Agent controls"
            className="grid shrink-0 grid-cols-2 border-b border-border"
          >
            <Tabs.Tab
              value="voice"
              className="px-3 py-2 text-muted-foreground data-active:border-b-2 data-active:border-agent data-active:text-agent"
            >
              Voice agent
            </Tabs.Tab>
            <Tabs.Tab
              value="latency"
              className="px-3 py-2 text-muted-foreground data-active:border-b-2 data-active:border-agent data-active:text-agent"
            >
              Model latency
            </Tabs.Tab>
          </Tabs.List>
          <Tabs.Panel
            value="voice"
            keepMounted
            className="min-h-0 space-y-4 overflow-y-auto pt-2 pb-1 [[hidden]]:hidden"
          >
            <Panel
              title="voice agent"
              status={
                <span
                  role="status"
                  className={ready ? "text-active" : "text-muted-foreground"}
                >
                  ● {status}
                </span>
              }
            >
              <div className="space-y-2 p-3 pt-4">
                <div
                  className="flex h-9 items-center justify-center"
                  aria-hidden="true"
                >
                  <AudioVisualizerBar
                    participantType="bot"
                    barOrigin="bottom"
                    barCount={40}
                    barWidth={3}
                    barGap={3}
                    barColor="--agent"
                    barMaxHeight={32}
                    barLineCap="square"
                  />
                </div>
                <p className="text-muted-foreground">
                  Speak naturally. Pause to finish your turn, or interrupt to
                  take it back.
                </p>
                <ConnectButton
                  onConnect={() => void connect()}
                  onDisconnect={() => void disconnect()}
                  disabled={busy}
                  className="h-9 w-full"
                />
                <UserAudioControl
                  disabled={!ready || micUnavailable}
                  className="h-9 w-full"
                />
                <ModelSelector />
                {micUnavailable && (
                  <p role="status" className="text-tool">
                    Microphone unavailable. You can type below, or allow
                    microphone access and reconnect.
                  </p>
                )}
                {error && (
                  <p role="alert" className="break-words text-inactive">
                    {error}
                  </p>
                )}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="mb-1.5 text-xs text-muted-foreground">
                      Microphone
                    </p>
                    <DeviceSelect kind="audioinput" className="w-full" />
                  </div>
                  <div>
                    <p className="mb-1.5 text-xs text-muted-foreground">
                      Speakers
                    </p>
                    <DeviceSelect kind="audiooutput" className="w-full" />
                  </div>
                </div>
              </div>
            </Panel>
            <NoisePanel />
          </Tabs.Panel>
          <Tabs.Panel
            value="latency"
            className="min-h-0 overflow-y-auto pt-2 pb-1"
          >
            <LatencyPanel />
          </Tabs.Panel>
        </Tabs.Root>
        <div className="grid min-h-0 min-w-0 grid-rows-[minmax(240px,1fr)_minmax(280px,1fr)] gap-4 pt-2 md:grid-rows-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <Panel
            title="live transcript"
            className="flex min-h-0 min-w-0 flex-col"
            status={
              <span className="text-xs text-muted-foreground">
                Voice + text
              </span>
            }
          >
            <div className="flex min-h-0 flex-1 flex-col p-3 pt-4">
              <Conversation />
              <div className="mt-2 flex items-center gap-2 border-t border-border pt-2">
                <TextInput />
              </div>
            </div>
          </Panel>
          <ComparisonPanel />
        </div>
      </main>
      <footer className="flex flex-wrap justify-between gap-2 text-[11px] text-muted-foreground">
        <span>SmallWebRTC · Silero VAD · Smart Turn</span>
        <span>Fine-tuned model: atc02</span>
      </footer>
      <BotAudioOutput />
    </div>
  )
}
export default function App() {
  return (
    <PipecatClientProvider client={client}>
      <Session />
    </PipecatClientProvider>
  )
}
