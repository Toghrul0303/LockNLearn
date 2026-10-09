"use client"

import type { PointerEvent } from "react"
import { type Components } from "react-markdown"
import { mathRemarkPlugins, mathRehypePlugins, texifyDeskMath } from "@/lib/markdown-math"
import { MathMarkdown } from "../math-markdown"

const inlineMathComponents: Components = {
  p: ({ children }) => <span>{children}</span>,
}

export function formatBoardValue(value: unknown): string {
  if (value == null) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>
    for (const key of ["value", "result", "text", "display", "formatted"]) {
      if (key in obj && obj[key] != null && typeof obj[key] !== "object") {
        return String(obj[key])
      }
    }
    try {
      return JSON.stringify(value)
    } catch {
      return ""
    }
  }
  return String(value)
}

export function BoardRichText({
  text,
  className,
  block = false,
}: {
  text: string
  className?: string
  block?: boolean
}) {
  const source = texifyDeskMath(text)
  if (block) {
    return (
      <MathMarkdown
        as="div"
        className={className ? `board-step-math ${className}` : "board-step-math"}
        remarkPlugins={mathRemarkPlugins}
        rehypePlugins={mathRehypePlugins}
      >
        {source}
      </MathMarkdown>
    )
  }
  return (
    <MathMarkdown
      as="span"
      className={className ? `math-inline ${className}` : "math-inline"}
      remarkPlugins={mathRemarkPlugins}
      rehypePlugins={mathRehypePlugins}
      components={inlineMathComponents}
    >
      {source}
    </MathMarkdown>
  )
}

export function onMathPointerDown(event: PointerEvent) {
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest("button, a, [data-board-action]")) return
  if (target.closest("mjx-container, .math-inline, [data-highlight-source]")) {
    event.stopPropagation()
  }
}
