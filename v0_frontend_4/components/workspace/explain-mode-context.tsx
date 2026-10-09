"use client"

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react"

type ExplainModeValue = {
  armed: boolean
  setArmed: (armed: boolean) => void
  disarm: () => void
}

const EXPLAIN_ARMED_CLASS = "explain-armed"

const ExplainModeContext = createContext<ExplainModeValue>({
  armed: false,
  setArmed: () => {},
  disarm: () => {},
})

export function syncExplainArmedClass(on: boolean) {
  document.documentElement.classList.toggle(EXPLAIN_ARMED_CLASS, on)
}

export function ExplainModeProvider({ children }: { children: ReactNode }) {
  const [armed, setArmedState] = useState(false)

  const setArmed = useCallback((next: boolean) => {
    setArmedState(next)
  }, [])

  const disarm = useCallback(() => {
    setArmedState(false)
  }, [])

  const value = useMemo(() => ({ armed, setArmed, disarm }), [armed, setArmed, disarm])
  return <ExplainModeContext.Provider value={value}>{children}</ExplainModeContext.Provider>
}

export function useExplainMode() {
  return useContext(ExplainModeContext)
}
