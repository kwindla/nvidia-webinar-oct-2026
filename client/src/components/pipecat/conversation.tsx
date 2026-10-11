import { usePipecatConversation } from "@pipecat-ai/client-react"
import { ConversationMessageItem } from "@/components/pipecat/conversation-message"

// DOM synchronization is confined to a React 19 callback ref. Follow new
// messages unless the reader scrolls up; release observers on unmount.
function observeTranscript(element: HTMLDivElement | null) {
  if (!element) return
  let following = true
  const onScroll = () => {
    following =
      element.scrollHeight - element.clientHeight - element.scrollTop < 48
  }
  const observer = new ResizeObserver(() => {
    if (following) element.scrollTop = element.scrollHeight
  })
  if (element.firstElementChild) observer.observe(element.firstElementChild)
  element.addEventListener("scroll", onScroll)
  return () => {
    observer.disconnect()
    element.removeEventListener("scroll", onScroll)
  }
}

export function Conversation() {
  const { messages } = usePipecatConversation()
  return (
    <div
      ref={observeTranscript}
      role="log"
      aria-label="Conversation"
      aria-live="polite"
      className="min-h-0 flex-1 overflow-y-auto"
    >
      <div className="flex flex-col gap-3 py-1">
        {messages.length ? (
          messages.map((message, index) => (
            <ConversationMessageItem key={index} message={message} />
          ))
        ) : (
          <div className="py-4 text-muted-foreground">
            <p className="mb-3 text-foreground">
              Your conversation starts here.
            </p>
            <p>Connect, allow your microphone, and speak naturally.</p>
            <p className="mt-3 text-xs">
              Try: “United four fifty two, turn left heading two seven zero.”
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
