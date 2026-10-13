import { PipecatClient } from "@pipecat-ai/client-js"
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport"
import { create } from "zustand"
import { useComparison } from "@/stores/comparison"
import { useMetrics } from "@/stores/metrics"
import { createSessionState, type SessionState } from "./session-state"
export type { ModelChoice } from "./session-state"

export const useSession = create<SessionState>(
  createSessionState({
    getClient: () => client,
    onNewSession: () => {
      useMetrics.getState().reset()
      useComparison.getState().reset()
    },
  })
)

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
      const generation = useSession.getState().generation
      void client
        .sendClientRequest("get-comparison", {}, 10000)
        .then((data) => {
          if (generation === useSession.getState().generation)
            useComparison.getState().ingest(data)
        })
        .catch(() => {})
    },
    onDisconnected: () => {
      useComparison.getState().freeze()
      useSession.getState().sessionEnded()
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
    useSession.getState().dispose()
    void client.disconnect().catch(() => {})
  })
}
