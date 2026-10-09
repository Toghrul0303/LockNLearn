export type CanvasOpTarget = "chat" | "canvas"

type CanvasOpBase = {
  /** Chat-only payloads must never update tldraw geometry. */
  target?: CanvasOpTarget
}

export type CanvasOp =
  | (CanvasOpBase & { op: "question"; prompt: string; source?: string; freshCard?: boolean })
  | (CanvasOpBase & { op: "step"; index: number; latex: string; branchFromId?: string; replaceLast?: boolean })
  | (CanvasOpBase & { op: "result"; value: string; summary?: string })
  | (CanvasOpBase & {
      op: "chart"
      chart_type: "line" | "bar" | "pie"
      title: string
      labels: string[]
      values: number[]
    })
  | (CanvasOpBase & {
      op: "diagram"
      title?: string
      width?: number
      height?: number
      elements: unknown[]
    })
  | (CanvasOpBase & { op: "figure"; image_url: string; index?: number })

export function normalizePrompt(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ")
}

function asString(value: unknown): string {
  if (value == null) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return ""
  }
}

export function isCanvasOp(value: unknown): value is CanvasOp {
  if (!value || typeof value !== "object") return false
  const rec = value as { op?: unknown; target?: unknown }
  if (rec.target === "chat") return false
  if (rec.target != null && rec.target !== "canvas") return false
  const op = rec.op
  if (op === "question") {
    return typeof (value as { prompt?: unknown }).prompt === "string"
  }
  if (op === "step") {
    const rec = value as { index?: unknown; latex?: unknown }
    return typeof rec.index === "number" && typeof rec.latex === "string"
  }
  if (op === "result") {
    return "value" in (value as object)
  }
  if (op === "chart") {
    const rec = value as { labels?: unknown; values?: unknown }
    return Array.isArray(rec.labels) && Array.isArray(rec.values)
  }
  if (op === "diagram") {
    return Array.isArray((value as { elements?: unknown }).elements)
  }
  if (op === "figure") {
    const url = (value as { image_url?: unknown }).image_url
    return typeof url === "string" && url.startsWith("data:image")
  }
  return false
}

/** Expand a legacy `desk_update` blob into sequential canvas ops. */
export function deskUpdateToCanvasOps(update: Record<string, unknown>): CanvasOp[] {
  const type = update.type
  if (type === "calculation") {
    const ops: CanvasOp[] = []
    const steps = Array.isArray(update.steps) ? update.steps : []
    steps.forEach((step, index) => {
      if (typeof step === "string" && step.trim()) {
        ops.push({ op: "step", index, latex: step })
      }
    })
    ops.push({
      op: "result",
      value: asString(update.value),
      summary: typeof update.summary === "string" ? update.summary : "",
    })
    return ops
  }
  if (type === "chart") {
    const chartType = update.chart_type
    if (chartType !== "line" && chartType !== "bar" && chartType !== "pie") return []
    return [
      {
        op: "chart",
        chart_type: chartType,
        title: typeof update.title === "string" ? update.title : "",
        labels: Array.isArray(update.labels) ? update.labels.map((label) => String(label)) : [],
        values: Array.isArray(update.values)
          ? update.values.map((value) => Number(value) || 0)
          : [],
      },
    ]
  }
  if (type === "diagram") {
    return [
      {
        op: "diagram",
        title: typeof update.title === "string" ? update.title : "",
        width: typeof update.width === "number" ? update.width : undefined,
        height: typeof update.height === "number" ? update.height : undefined,
        elements: Array.isArray(update.elements) ? update.elements : [],
      },
    ]
  }
  return []
}
