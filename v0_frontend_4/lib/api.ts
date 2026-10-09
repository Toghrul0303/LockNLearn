/**
 * Client for the FastAPI `/submit_stream` endpoint.
 *
 * The backend streams Server-Sent Events, one JSON payload per tick:
 *   data: {"chat_message": "...", "desk_update": {...} | null, "active_problem_update": {...}, "canvas_op": {...}, "canvas_anchor_id": "..."}
 *
 * This module owns the fetch + streaming-parse logic so UI components can
 * just supply callbacks for the independent payload channels.
 */

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/+$/, "")

export const SUBMIT_STREAM_URL = `${API_BASE_URL}/submit_stream`

export type StudyMode = "detailed" | "socratic"

export type StreamChatParams = {
  message: string
  threadId: string
  mode: StudyMode
  uiLanguage?: string
  document?: File | null
  canvasAnchorId?: string
  canvasRequestId?: string
  /** Parent board-step id for a Why/How lateral explanation branch. */
  canvasBranchFromId?: string
  /** Alt+selection origin. Chat stays in the sidebar; desk draws steps. */
  explainSource?: "chat" | "desk"
  userId?: string
  accessToken?: string
  documentPath?: string
  /** Called for every non-empty `chat_message` chunk, in arrival order. */
  onChatChunk: (chunk: string) => void
  /** Called only when a tick carries a non-null structured `desk_update`. */
  onDeskUpdate: (update: Record<string, unknown>, meta?: { canvasAnchorId?: string }) => void
  /** Incremental canvas ops (preferred over a batched calculation blob). */
  onCanvasOp?: (op: Record<string, unknown>, meta?: { canvasAnchorId?: string }) => void
  /** Called when the composer emits the Active Problem header for this turn. */
  onActiveProblemUpdate?: (problem: Record<string, unknown>) => void
  /** Terminal close — graph finished, errored, or sent an explicit `done` tick. */
  onDone?: (meta?: { canvasAnchorId?: string }) => void
  /** English pipeline stage while the answer bubble is still empty. */
  onStatus?: (status: string) => void
  signal?: AbortSignal
}

/**
 * Posts the chat turn to the backend and incrementally parses the SSE
 * response stream, dispatching each tick's `chat_message` / `desk_update`
 * / `canvas_op` / `active_problem_update` to the provided callbacks as soon as it arrives.
 */
export async function streamChatCompletion({
  message,
  threadId,
  mode,
  uiLanguage,
  document,
  canvasAnchorId,
  canvasRequestId,
  canvasBranchFromId,
  explainSource,
  userId,
  accessToken,
  documentPath,
  onChatChunk,
  onDeskUpdate,
  onCanvasOp,
  onActiveProblemUpdate,
  onDone,
  onStatus,
  signal,
}: StreamChatParams): Promise<void> {
  const formData = new FormData()
  formData.append("message", message)
  formData.append("thread_id", threadId)
  formData.append("mode", mode)
  if (uiLanguage) {
    formData.append("ui_language", uiLanguage)
  }
  if (canvasAnchorId) {
    formData.append("canvas_anchor_id", canvasAnchorId)
  }
  if (canvasRequestId) {
    formData.append("canvas_request_id", canvasRequestId)
  }
  if (canvasBranchFromId) {
    formData.append("canvas_branch_from_id", canvasBranchFromId)
  }
  if (explainSource) {
    formData.append("explain_source", explainSource)
  }
  if (userId) formData.append("user_id", userId)
  if (accessToken) formData.append("access_token", accessToken)
  if (documentPath) formData.append("document_path", documentPath)
  if (document) {
    formData.append("document", document)
  }

  const response = await fetch(SUBMIT_STREAM_URL, {
    method: "POST",
    body: formData,
    signal,
  })

  if (!response.ok) {
    throw new Error(`Chat request failed with status ${response.status}`)
  }
  if (!response.body) {
    throw new Error("Chat response did not include a readable stream body.")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line.
    let frameEnd = buffer.indexOf("\n\n")
    while (frameEnd !== -1) {
      const rawFrame = buffer.slice(0, frameEnd)
      buffer = buffer.slice(frameEnd + 2)
      dispatchSseFrame(rawFrame, onChatChunk, onDeskUpdate, onActiveProblemUpdate, onCanvasOp, onDone, onStatus)
      frameEnd = buffer.indexOf("\n\n")
    }
  }

  // Flush a trailing frame that wasn't terminated by a final blank line.
  if (buffer.trim()) {
    dispatchSseFrame(buffer, onChatChunk, onDeskUpdate, onActiveProblemUpdate, onCanvasOp, onDone, onStatus)
  }
}

function dispatchSseFrame(
  rawFrame: string,
  onChatChunk: (chunk: string) => void,
  onDeskUpdate: (update: Record<string, unknown>, meta?: { canvasAnchorId?: string }) => void,
  onActiveProblemUpdate?: (problem: Record<string, unknown>) => void,
  onCanvasOp?: (op: Record<string, unknown>, meta?: { canvasAnchorId?: string }) => void,
  onDone?: (meta?: { canvasAnchorId?: string }) => void,
  onStatus?: (status: string) => void,
) {
  for (const line of rawFrame.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed.startsWith("data:")) continue

    const jsonText = trimmed.slice("data:".length).trim()
    if (!jsonText) continue

    // Defensive: never let a malformed/partial JSON line crash the stream.
    // Plain-text or unparsable ticks are simply skipped, which is what
    // keeps "desk_update: null" (or a missing field) from ever throwing.
    let payload: {
      chat_message?: string
      desk_update?: Record<string, unknown> | null
      active_problem_update?: Record<string, unknown> | null
      canvas_op?: Record<string, unknown> | null
      canvas_ops?: unknown
      canvas_anchor_id?: string | null
      done?: boolean
      status?: string
    }
    try {
      payload = JSON.parse(jsonText)
    } catch {
      continue
    }

    const meta = payload.canvas_anchor_id
      ? { canvasAnchorId: payload.canvas_anchor_id }
      : undefined

    if (payload.chat_message) {
      onChatChunk(payload.chat_message)
    }
    if (payload.canvas_op != null) {
      onCanvasOp?.(payload.canvas_op, meta)
    }
    if (Array.isArray(payload.canvas_ops)) {
      for (const op of payload.canvas_ops) {
        if (op && typeof op === "object") onCanvasOp?.(op as Record<string, unknown>, meta)
      }
    }
    if (payload.desk_update != null) {
      onDeskUpdate(payload.desk_update, meta)
    }
    if (payload.active_problem_update != null) {
      onActiveProblemUpdate?.(payload.active_problem_update)
    }
    if (payload.status) {
      onStatus?.(payload.status)
    }
    if (payload.done) {
      onDone?.(meta)
    }
  }
}

export type StrugglePayload = {
  thread_id: string
  question_id: string
  topic: string
  excerpt: string
  formula?: string
}

/** Writes a Needs Review record into persistent struggle memory. */
export async function recordStruggle(payload: StrugglePayload): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/memory/struggle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error(`Struggle save failed (${response.status})`)
  }
}

export async function nameSavedFormula(params: {
  formulaId: string
  formula: string
  userId?: string
  accessToken?: string
}): Promise<string | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/name_formula`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        formula: params.formula,
        formula_id: params.formulaId,
        user_id: params.userId || "",
        access_token: params.accessToken || "",
      }),
    })
    if (!response.ok) return null
    const data = (await response.json()) as { context?: unknown }
    const name = typeof data.context === "string" ? data.context.trim() : ""
    return name || null
  } catch {
    return null
  }
}
