"use client"

import { createContext, useContext } from "react"

type ModeContextValue = {
  dark: boolean
  toggleTheme: () => void
}

export const ModeContext = createContext<ModeContextValue>({
  dark: false,
  toggleTheme: () => {},
})

export function useMode() {
  return useContext(ModeContext)
}
