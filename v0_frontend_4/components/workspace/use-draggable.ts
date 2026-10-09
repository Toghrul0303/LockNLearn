"use client"

import { useCallback, useEffect, useRef, useState } from "react"

type Point = { x: number; y: number }

const MARGIN = 8

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

/**
 * Makes an element free-dragging via pointer events.
 *
 * The element keeps its default CSS anchor (e.g. `fixed bottom-4 left-4`)
 * until the first drag, after which it is positioned with fixed `left`/`top`
 * viewport coordinates. Position is clamped to stay on-screen and re-clamped
 * when the viewport resizes. Works with any default anchor corner.
 */
export function useDraggable() {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<Point | null>(null)
  const [dragging, setDragging] = useState(false)
  const origin = useRef<{ pointer: Point; start: Point } | null>(null)

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    // Only start dragging with the primary (usually left) button.
    if (e.button !== 0) return
    const el = ref.current
    if (!el) return
    e.preventDefault()
    const rect = el.getBoundingClientRect()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    origin.current = {
      pointer: { x: e.clientX, y: e.clientY },
      start: { x: rect.left, y: rect.top },
    }
    // Lock to the current rendered position so the drag starts seamlessly.
    setPos({ x: rect.left, y: rect.top })
    setDragging(true)
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!origin.current) return
    const dx = e.clientX - origin.current.pointer.x
    const dy = e.clientY - origin.current.pointer.y
    let nextX = origin.current.start.x + dx
    let nextY = origin.current.start.y + dy

    const el = ref.current
    if (el) {
      const rect = el.getBoundingClientRect()
      nextX = clamp(nextX, MARGIN, window.innerWidth - rect.width - MARGIN)
      nextY = clamp(nextY, MARGIN, window.innerHeight - rect.height - MARGIN)
    }
    setPos({ x: nextX, y: nextY })
  }, [])

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    origin.current = null
    setDragging(false)
    try {
      ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* pointer may already be released */
    }
  }, [])

  // Keep the element on-screen if the viewport shrinks.
  useEffect(() => {
    const onResize = () => {
      const el = ref.current
      setPos((prev) => {
        if (!prev || !el) return prev
        const rect = el.getBoundingClientRect()
        return {
          x: clamp(prev.x, MARGIN, window.innerWidth - rect.width - MARGIN),
          y: clamp(prev.y, MARGIN, window.innerHeight - rect.height - MARGIN),
        }
      })
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  const style: React.CSSProperties | undefined = pos
    ? { left: pos.x, top: pos.y, right: "auto", bottom: "auto" }
    : undefined

  return {
    ref,
    style,
    dragging,
    handleProps: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
    },
  }
}
