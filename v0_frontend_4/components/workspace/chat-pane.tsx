"use client"

import { useState } from "react"
import { AttachmentChip } from "./attachment-chip"
import { MathMarkdown } from "./math-markdown"
import { TutorAvatar } from "./tutor-avatar"
import { useStudyMode } from "./study-mode-context"

type Bubble = { id: string; role: "user" | "ai"; content: string }

const STARTER: Bubble[] = [
  {
    id: "hello",
    role: "ai",
    content: "Bring a problem over and we can work it on the board.",
  },
]

export function ChatPane() {
  const { studyMode } = useStudyMode()
  const [messages, setMessages] = useState<Bubble[]>(STARTER)
  const [draft, setDraft] = useState("")

  const send = () => {
    const text = draft.trim()
    if (!text) return
    setMessages((current) => [
      ...current,
      { id: `u-${current.length}`, role: "user", content: text },
      {
        id: `a-${current.length}`,
        role: "ai",
        content: `Noted in ${studyMode.shortLabel} mode. The tutor reply will show up in this thread.`,
      },
    ])
    setDraft("")
  }

  return (
    <section className="flex h-full min-h-0 w-[380px] shrink-0 flex-col border-l border-[#e8e4ee] bg-white">
      <header className="flex items-center gap-2 border-b border-[#efedf2] px-4 py-3">
        <TutorAvatar />
        <div>
          <p className="text-sm font-semibold text-[#353042]">AI Tutor</p>
          <p className="text-[11px] text-[#797483]">{studyMode.label}</p>
        </div>
      </header>
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.map((message) => (
          <div key={message.id} className={message.role === "user" ? "ml-8" : "mr-6"}>
            {message.role === "user" ? <AttachmentChip name="You" /> : null}
            <div className={`mt-1 rounded-2xl px-3 py-2 ${message.role === "user" ? "bg-[#fff0f8]" : "bg-[#f7f6f9]"}`}>
              <MathMarkdown text={message.content} />
            </div>
          </div>
        ))}
      </div>
      <form
        className="border-t border-[#efedf2] p-3"
        onSubmit={(event) => {
          event.preventDefault()
          send()
        }}
      >
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          rows={3}
          placeholder="Ask about a step..."
          className="w-full resize-none rounded-2xl border border-[#e5e1e9] px-3 py-2 text-sm outline-none focus:border-[#ba2b8a]"
        />
        <button type="submit" className="mt-2 w-full rounded-full bg-[#ba2b8a] py-2 text-sm font-semibold text-white">
          Send
        </button>
      </form>
    </section>
  )
}

export function ChatCollapseFab() {
  return null
}
