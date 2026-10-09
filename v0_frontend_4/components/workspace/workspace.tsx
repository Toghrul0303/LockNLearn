"use client"

import { useCallback, useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { ModeContext } from "./mode-context"
import { TopNavbar } from "./top-navbar"
import { LeftSidebar } from "./left-sidebar"
import { Desk } from "./desk"
import { ChatCollapseFab, ChatPane } from "./chat-pane"
import { TaskTrackerProvider } from "./task-tracker-context"
import { DeskProvider } from "./desk-context"
import { SessionProvider } from "./session-context"
import { SessionSliceSync } from "./session-slice-sync"
import { StudyModeProvider } from "./study-mode-context"
import { ExplainModeProvider } from "./explain-mode-context"
import { WhiteboardProvider } from "./canvas/whiteboard-context"

const THEME_KEY = "locknlearn.theme"
const CHAT_WIDTH_KEY = "locknlearn.chatWidth"
const CHAT_WIDTH_DEFAULT = 400
const CHAT_WIDTH_MIN = 360

function clampChatWidth(value: number) {
  const max = Math.max(CHAT_WIDTH_MIN, Math.round(window.innerWidth * 0.33))
  return Math.min(Math.max(Math.round(value), CHAT_WIDTH_MIN), max)
}

export function Workspace() {
  const [dark, setDark] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [chatCollapsed, setChatCollapsed] = useState(false)
  const [chatWidth, setChatWidth] = useState(CHAT_WIDTH_DEFAULT)

  useEffect(() => {
    setDark(window.localStorage.getItem(THEME_KEY) === "dark")
    const stored = Number(window.localStorage.getItem(CHAT_WIDTH_KEY))
    if (Number.isFinite(stored) && stored > 0) setChatWidth(clampChatWidth(stored))
  }, [])

  useEffect(() => {
    const onResize = () => setChatWidth((width) => clampChatWidth(width))
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  const setChatWidthLive = useCallback((value: number) => {
    setChatWidth(clampChatWidth(value))
  }, [])

  const commitChatWidth = useCallback((value: number) => {
    const next = clampChatWidth(value)
    setChatWidth(next)
    window.localStorage.setItem(CHAT_WIDTH_KEY, String(next))
  }, [])

  const toggleTheme = useCallback(() => {
    setDark((prev) => {
      const next = !prev
      window.localStorage.setItem(THEME_KEY, next ? "dark" : "light")
      return next
    })
  }, [])

  return (
    <ModeContext.Provider value={{ dark, toggleTheme }}>
      <SessionProvider>
      <TaskTrackerProvider>
      <DeskProvider>
      <WhiteboardProvider>
      <StudyModeProvider>
      <ExplainModeProvider>
        <SessionSliceSync />
        <div className={cn("h-dvh w-full overflow-hidden", dark && "dark")}>
        <div className="relative flex h-full flex-col bg-background text-foreground">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0"
            style={{
              backgroundImage:
                "radial-gradient(60rem 40rem at 85% -10%, color-mix(in oklab, var(--brand-purple) 14%, transparent), transparent), radial-gradient(50rem 40rem at -10% 110%, color-mix(in oklab, var(--brand-red) 12%, transparent), transparent)",
            }}
          />

          <div className="relative z-10 flex h-full flex-col">
            <TopNavbar />

            <div className="flex min-h-0 flex-1">
              <LeftSidebar
                collapsed={sidebarCollapsed}
                onToggle={() => setSidebarCollapsed((c) => !c)}
              />

              <main className="relative flex min-w-0 flex-1">
                <Desk />
              </main>

              <div
                className={cn("shrink-0", chatCollapsed && "w-0")}
                style={chatCollapsed ? undefined : { width: chatWidth }}
              >
                <ChatPane
                  collapsed={chatCollapsed}
                  chatWidth={chatWidth}
                  onChatWidthChange={setChatWidthLive}
                  onChatWidthCommit={commitChatWidth}
                  onToggle={() => setChatCollapsed((c) => !c)}
                />
              </div>
            </div>
            <ChatCollapseFab
              collapsed={chatCollapsed}
              onExpand={() => setChatCollapsed(false)}
            />
          </div>
        </div>
      </div>
      </ExplainModeProvider>
      </StudyModeProvider>
      </WhiteboardProvider>
      </DeskProvider>
      </TaskTrackerProvider>
      </SessionProvider>
    </ModeContext.Provider>
  )
}
