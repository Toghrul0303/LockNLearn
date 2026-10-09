"use client"

import { createContext, useContext, useMemo, useState, type ReactNode } from "react"

export type BoardShape = {
  id: string
  kind: "card" | "figure" | "chart" | "calculator"
  x: number
  y: number
  title: string
  body: string
}

type WhiteboardValue = {
  shapes: BoardShape[]
}

const WhiteboardContext = createContext<WhiteboardValue>({ shapes: [] })

export function WhiteboardProvider({ children }: { children: ReactNode }) {
  const [shapes] = useState<BoardShape[]>([
    {
      id: "card-1",
      kind: "card",
      x: 48,
      y: 48,
      title: "Torque",
      body: "τ = ½ M R² α",
    },
  ])
  const value = useMemo(() => ({ shapes }), [shapes])
  return <WhiteboardContext.Provider value={value}>{children}</WhiteboardContext.Provider>
}

export function useWhiteboard() {
  return useContext(WhiteboardContext)
}
