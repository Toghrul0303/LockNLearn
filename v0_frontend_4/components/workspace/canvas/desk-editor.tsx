"use client"

import { BoardShapes } from "./board-shapes"
import { useWhiteboard } from "./whiteboard-context"

export function DeskEditor() {
  const { shapes } = useWhiteboard()
  return (
    <div className="relative min-h-[520px] flex-1 overflow-hidden rounded-[22px] border border-[#e2dfe8] bg-[#f8f8fa]">
      <BoardShapes shapes={shapes} />
    </div>
  )
}
