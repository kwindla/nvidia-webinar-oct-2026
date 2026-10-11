import {
  useConversationContext,
  usePipecatClient,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { useSession } from "@/stores/session"

export function TextInput() {
  const client = usePipecatClient()
  const state = usePipecatClientTransportState()
  const { injectMessage } = useConversationContext()
  const [value, setValue] = useState("")
  const [sending, setSending] = useState(false)
  const disabled = state !== "ready" || sending
  return (
    <form
      className="flex min-w-0 flex-1 gap-2"
      onSubmit={async (event) => {
        event.preventDefault()
        const text = value.trim()
        if (!text || disabled || !client) return
        setSending(true)
        try {
          // Establish the user turn before a fast local response can arrive.
          injectMessage({
            role: "user",
            parts: [{ text, final: true, createdAt: new Date().toISOString() }],
          })
          await client.sendText(text)
          setValue("")
        } catch {
          useSession.setState({
            error: "Message could not be sent. Please try again.",
          })
        } finally {
          setSending(false)
        }
      }}
    >
      <input
        aria-label="Message"
        placeholder="Or type a message…"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
        className="min-w-0 flex-1 border border-border bg-background px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-active disabled:opacity-50"
      />
      <Button
        type="submit"
        variant="outline"
        disabled={disabled || !value.trim()}
      >
        Send
      </Button>
    </form>
  )
}
