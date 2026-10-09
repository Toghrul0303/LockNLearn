"use client"

import { createContext, useContext, useMemo, useState, type ReactNode } from "react"

export type DeskCard = {
  id: string
  title: string
  body: string
}

type DeskContextValue = {
  cards: DeskCard[]
}

const DeskContext = createContext<DeskContextValue>({ cards: [] })

const SAMPLE_CARDS: DeskCard[] = [
  {
    id: "sample-question",
    title: "Question",
    body: "A disc of mass M and radius R starts from rest. Find the torque that produces angular acceleration α.",
  },
  {
    id: "sample-step",
    title: "Step",
    body: "τ = Iα, and for a solid disc I = ½MR².",
  },
]

export function DeskProvider({ children }: { children: ReactNode }) {
  const [cards] = useState(SAMPLE_CARDS)
  const value = useMemo(() => ({ cards }), [cards])
  return <DeskContext.Provider value={value}>{children}</DeskContext.Provider>
}

export function useDesk() {
  return useContext(DeskContext)
}
