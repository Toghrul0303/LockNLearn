"use client"

import type { RefObject } from "react"
import { PomodoroTimer } from "./pomodoro-timer"

/** Pomodoro dock, centered on the Desk canvas. The music player is deferred. */
export function BottomDock({
  constraintsRef,
}: {
  constraintsRef?: RefObject<HTMLDivElement | null>
}) {
  return (
    <div className="absolute bottom-4 left-1/2 z-40 flex -translate-x-1/2 items-center">
      <PomodoroTimer dragConstraints={constraintsRef} />
    </div>
  )
}
