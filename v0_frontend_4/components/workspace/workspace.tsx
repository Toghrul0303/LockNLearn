"use client"

import { useCallback, useState } from "react"
import { Desk } from "./desk"
import { DeskProvider } from "./desk-context"
import { ChatPane } from "./chat-pane"
import { LeftSidebar } from "./left-sidebar"
import { ModeContext } from "./mode-context"
import { StudyModeProvider } from "./study-mode-context"
import { TopNavbar } from "./top-navbar"
import { WhiteboardProvider } from "./canvas/whiteboard-context"

export function Workspace() {
  const [dark, setDark] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const toggleTheme = useCallback(() => setDark((value) => !value), [])

  return (
    <ModeContext.Provider value={{ dark, toggleTheme }}>
      <DeskProvider>
        <WhiteboardProvider>
          <StudyModeProvider>
            <div className={dark ? "dark" : ""}>
              <div className="flex h-dvh flex-col bg-background text-foreground">
                <TopNavbar />
                <div className="flex min-h-0 flex-1">
                  <LeftSidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)} />
                  <Desk />
                  <ChatPane />
                </div>
              </div>
            </div>
          </StudyModeProvider>
        </WhiteboardProvider>
      </DeskProvider>
    </ModeContext.Provider>
  )
}
