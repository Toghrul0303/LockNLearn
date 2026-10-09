"use client"

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react"
import { TOOL_MODES, type ToolMode } from "./data"

type StudyModeContextValue = {
  studyMode: ToolMode
  setStudyMode: (mode: ToolMode | string) => void
}

const StudyModeContext = createContext<StudyModeContextValue>({
  studyMode: TOOL_MODES[0],
  setStudyMode: () => {},
})

export function StudyModeProvider({ children }: { children: ReactNode }) {
  const [studyMode, setStudyModeState] = useState<ToolMode>(TOOL_MODES[0])

  const setStudyMode = useCallback((mode: ToolMode | string) => {
    const next =
      typeof mode === "string" ? TOOL_MODES.find((item) => item.id === mode) : mode
    if (next) setStudyModeState(next)
  }, [])

  const value = useMemo(() => ({ studyMode, setStudyMode }), [studyMode, setStudyMode])

  return <StudyModeContext.Provider value={value}>{children}</StudyModeContext.Provider>
}

export function useStudyMode() {
  return useContext(StudyModeContext)
}
