import { createShapeId, toRichText, type Editor, type TLShapeId } from "tldraw"
import { collisionShift } from "./canvas-layout"

type DiagramEl = {
  type?: string
  x?: number
  y?: number
  x1?: number
  y1?: number
  x2?: number
  y2?: number
  cx?: number
  cy?: number
  r?: number
  width?: number
  height?: number
  start_deg?: number
  end_deg?: number
  label?: string
  text?: string
  style?: "solid" | "dashed"
}

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function createLine(
  editor: Editor,
  origin: { x: number; y: number },
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  vector: boolean,
  dashed: boolean,
): TLShapeId {
  const id = createShapeId()
  editor.createShape({
    id,
    type: "arrow",
    x: origin.x + x1,
    y: origin.y + y1,
    props: {
      start: { x: 0, y: 0 },
      end: { x: x2 - x1, y: y2 - y1 },
      arrowheadStart: "none",
      arrowheadEnd: vector ? "arrow" : "none",
      dash: dashed ? "dashed" : "draw",
    },
  })
  return id
}

function createLabel(editor: Editor, x: number, y: number, text: string): TLShapeId | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  const id = createShapeId()
  try {
    editor.createShape({
      id,
      type: "text",
      x,
      y,
      props: {
        richText: toRichText(trimmed),
        size: "s",
        autoSize: true,
      },
    })
    return id
  } catch {
    return null
  }
}

export function spawnDiagramPrimitives(
  editor: Editor,
  payload: { title?: string; width?: number; height?: number; elements?: unknown[] },
  origin: { x: number; y: number },
): TLShapeId[] {
  const elements = Array.isArray(payload.elements) ? (payload.elements as DiagramEl[]) : []
  const width = num(payload.width, 400)
  const height = num(payload.height, 300)
  const shifted = collisionShift(editor, origin.x, origin.y, width, height, [])
  const ids: TLShapeId[] = []

  if (payload.title) {
    const titleId = createLabel(editor, shifted.x, shifted.y - 28, payload.title)
    if (titleId) ids.push(titleId)
  }

  for (const el of elements) {
    const dashed = el.style === "dashed"
    const type = el.type
    if (type === "rect") {
      const id = createShapeId()
      const w = Math.max(8, num(el.width, 60))
      const h = Math.max(8, num(el.height, 40))
      editor.createShape({
        id,
        type: "geo",
        x: shifted.x + num(el.x),
        y: shifted.y + num(el.y),
        props: {
          geo: "rectangle",
          w,
          h,
          fill: "none",
          dash: dashed ? "dashed" : "draw",
        },
      })
      ids.push(id)
      const labelId = createLabel(
        editor,
        shifted.x + num(el.x) + w / 2 - 12,
        shifted.y + num(el.y) + h / 2 - 10,
        el.label || "",
      )
      if (labelId) ids.push(labelId)
      continue
    }
    if (type === "circle") {
      const r = Math.max(6, num(el.r, 20))
      const id = createShapeId()
      editor.createShape({
        id,
        type: "geo",
        x: shifted.x + num(el.cx) - r,
        y: shifted.y + num(el.cy) - r,
        props: {
          geo: "ellipse",
          w: r * 2,
          h: r * 2,
          fill: "none",
          dash: dashed ? "dashed" : "draw",
        },
      })
      ids.push(id)
      const labelId = createLabel(editor, shifted.x + num(el.cx), shifted.y + num(el.cy) - r - 18, el.label || "")
      if (labelId) ids.push(labelId)
      continue
    }
    if (type === "line" || type === "vector") {
      ids.push(
        createLine(
          editor,
          shifted,
          num(el.x1),
          num(el.y1),
          num(el.x2),
          num(el.y2),
          type === "vector",
          dashed,
        ),
      )
      const labelId = createLabel(
        editor,
        shifted.x + (num(el.x1) + num(el.x2)) / 2,
        shifted.y + (num(el.y1) + num(el.y2)) / 2 - 14,
        el.label || "",
      )
      if (labelId) ids.push(labelId)
      continue
    }
    if (type === "arc") {
      const r = Math.max(6, num(el.r, 40))
      const id = createShapeId()
      editor.createShape({
        id,
        type: "geo",
        x: shifted.x + num(el.cx, num(el.x)) - r,
        y: shifted.y + num(el.cy, num(el.y)) - r,
        props: {
          geo: "ellipse",
          w: r * 2,
          h: r * 2,
          fill: "none",
          dash: dashed ? "dashed" : "draw",
        },
      })
      ids.push(id)
      continue
    }
    if (type === "text") {
      const labelId = createLabel(
        editor,
        shifted.x + num(el.x, num(el.cx)),
        shifted.y + num(el.y, num(el.cy)),
        el.text || el.label || "",
      )
      if (labelId) ids.push(labelId)
    }
  }

  if (ids.length === 0) return ids
  if (ids.length > 1) {
    try {
      editor.groupShapes(ids)
    } catch {
      /* grouping is optional */
    }
  }
  return ids
}
