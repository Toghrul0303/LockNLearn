"use client"

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import type { Editor, TLShapeId } from "tldraw"
import {
  isDeskCardShape,
  listDeskItemsFromEditor,
  readDeskItem,
  serializeDeskItem,
  deskCardShapeId,
} from "./canvas/desk-card-shape"
import {
  BOARD_BRANCH_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_QUESTION_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_STEP_TYPE,
  asShapeIdList,
  isBoardChartShape,
  isBoardQuestionShape,
  isBoardStepShape,
} from "./canvas/board-constants"
import {
  applyClusterVisibility,
  beginSolveSession,
  clearSolution,
  focusOnShape,
  groupQuestionCluster,
  reflowMainColumn,
  setQuestionStatus,
  setSolutionHidden,
  spawnChartNode,
  spawnFigureNode,
  spawnQuestionShape,
  spawnResultNode,
  spawnStepNode,
  updateLastMainColumnStep,
  syncSessionToQuestion,
  type CanvasSolveSession,
  QUESTION_H,
  QUESTION_W,
} from "./canvas/canvas-layout"
import { deskUpdateToCanvasOps, isCanvasOp, normalizePrompt, type CanvasOp } from "./canvas/canvas-ops"
import { migrateDeskCards, spawnLegacyDeskItems } from "./canvas/migrate-desk-cards"
import { spawnDiagramPrimitives } from "./canvas/spawn-diagram"
import { formatBoardValue } from "./canvas/board-math"
import { useTaskTracker } from "./task-tracker-context"
import {
  isolateProblemStem,
  isBookNavigationPrompt,
  isCoachSpeakPrompt,
  isPlaceholderQuestionPrompt,
  isRenderableQuestionPrompt,
  isSubstantialProblemStem,
  sanitizeCanvasPrompt,
  stemsAreSameProblem,
} from "@/lib/parse-task-assignment"

export type ChartDeskUpdate = {
  type: "chart"
  chart_type: "line" | "bar" | "pie"
  title: string
  labels: string[]
  values: number[]
}

export type CalculationDeskUpdate = {
  type: "calculation"
  value: number | string | Record<string, unknown>
  summary: string
  /** Optional step-by-step derivation — rendered as bound board-step nodes. */
  steps?: string[]
}

export type DiagramPrimitiveType = "rect" | "line" | "vector" | "circle" | "arc" | "text"

export type DiagramElement = {
  type: DiagramPrimitiveType
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

export type DiagramDeskUpdate = {
  type: "diagram"
  title: string
  width?: number
  height?: number
  summary?: string
  elements: DiagramElement[]
}

export type DeskUpdate = ChartDeskUpdate | CalculationDeskUpdate | DiagramDeskUpdate

export type DeskItemMeta = {
  headerTitle?: string
  headerDescription?: string
  bookmarked?: boolean
}

export type DeskItem = DeskUpdate & { id: string } & DeskItemMeta

export type WhiteboardMeta = {
  id: string
  name: string
  /** Set after a successful cloud upload. Compared with the IndexedDB clock. */
  cloudSavedAt?: number
}

export type Drawer = "calculator" | null

export type SolveMode = "detailed" | "socratic-steps"

export type SolveHandler = (
  prompt: string,
  shapeId: string,
  requestId: string,
  mode: SolveMode,
) => void

/** Two-line Active Problem header: `title` is source + problem, `description` is topic/question. */
export type ActiveProblem = {
  title: string
  description: string
  source?: string
  topic?: string
}

type PendingOp = {
  op: CanvasOp
  anchorId?: string
  requestId?: string
  trackerQuestionId?: string
}

type CanvasOpMeta = {
  canvasAnchorId?: string
  requestId?: string
  trackerQuestionId?: string
  /** Tracker chapter label used when the model only sent "Typed problem". */
  cardTitle?: string
}

type DeskContextValue = {
  /** Projected artifacts (legacy cards + board charts) for session sync / save-graph. */
  deskItems: DeskItem[]
  /** Validates and applies a raw `desk_update` object from the SSE stream. */
  pushDeskUpdate: (update: Record<string, unknown>, meta?: CanvasOpMeta) => string | null
  applyCanvasOp: (op: CanvasOp, meta?: CanvasOpMeta) => void
  spawnQuestion: (prompt: string, source?: string) => string | null
  /** Currently active board-question card (if it still exists) — lets a
   * Socratic-mode follow-up turn append to the same card instead of
   * spawning a new independent one. */
  getActiveQuestionId: () => string | null
  /** Pan to the board card created for this Task Tracker question, when one exists. */
  focusTrackerQuestion: (trackerQuestionId: string) => void
  /** Question stem stored on that board card, or an empty string. */
  readTrackerQuestionPrompt: (trackerQuestionId: string) => string
  /** Painted + measured question card — flushes any steps/results queued behind it. */
  notifyQuestionMeasured: (questionId: string) => void
  requestSolve: (shapeId: string, opts?: { mode?: SolveMode }) => void
  /** Solved cluster present → toggle its `solutionHidden` flag; idle → kick off a solve. */
  toggleSolution: (shapeId: string) => void
  completeSolve: (requestId?: string, shapeId?: string, status?: "idle" | "solved" | "error") => void
  registerSolveHandler: (handler: SolveHandler | null) => void
  removeDeskItem: (id: string) => void
  patchDeskItem: (id: string, patch: Partial<DeskItemMeta>) => void
  /** One-shot header backfill for cards created in the current turn. */
  stampHeadersOnItems: (ids: string[], title: string, description: string) => void
  clearDeskItems: () => void
  /** Replace the whole Desk pile (session hydrate / draft reset / legacy migration). */
  replaceDeskItems: (items: DeskItem[]) => void
  /** Drop queued creates/replaces so a snapshot hydrate is not overwritten. */
  resetCanvasQueue: () => void
  setEditor: (editor: Editor | null) => void
  /** Shared so the tldraw toolbar calculator button and the canvas overlay stay in sync. */
  drawer: Drawer
  toggleDrawer: (id: Exclude<Drawer, null>) => void
  closeDrawer: () => void
  /** The whiteboard header — set from backend `active_problem_update` (or a Desk-card fallback). */
  activeProblem: ActiveProblem | null
  setActiveProblem: (problem: ActiveProblem | null) => void
}

const DeskContext = createContext<DeskContextValue>({
  deskItems: [],
  pushDeskUpdate: () => null,
  applyCanvasOp: () => {},
  spawnQuestion: () => null,
  getActiveQuestionId: () => null,
  focusTrackerQuestion: () => {},
  readTrackerQuestionPrompt: () => "",
  notifyQuestionMeasured: () => {},
  requestSolve: () => {},
  toggleSolution: () => {},
  completeSolve: () => {},
  registerSolveHandler: () => {},
  removeDeskItem: () => {},
  patchDeskItem: () => {},
  stampHeadersOnItems: () => {},
  clearDeskItems: () => {},
  replaceDeskItems: () => {},
  resetCanvasQueue: () => {},
  setEditor: () => {},
  drawer: null,
  toggleDrawer: () => {},
  closeDrawer: () => {},
  activeProblem: null,
  setActiveProblem: () => {},
})

/** tldraw shape HTML can paint during HMR before DeskContext is current.
 * Board measure calls this module listener instead of `useDesk()` so a
 * stale/missing context value cannot swallow the queued step flush. */
const questionMeasureGate = {
  notify: (_questionId: string) => {},
}

export function notifyQuestionPainted(questionId: string) {
  questionMeasureGate.notify(questionId)
}

export function isDeskUpdate(value: Record<string, unknown>): value is DeskUpdate {
  if (!value || typeof value !== "object") return false

  if (value.type === "chart") {
    return Array.isArray((value as ChartDeskUpdate).labels) && Array.isArray((value as ChartDeskUpdate).values)
  }
  if (value.type === "calculation") {
    return "value" in value
  }
  if (value.type === "diagram") {
    return Array.isArray((value as DiagramDeskUpdate).elements)
  }
  return false
}

export function isActiveProblemUpdate(value: Record<string, unknown>): value is ActiveProblem {
  return typeof value?.title === "string" && value.title.trim().length > 0
}

function projectDeskItems(editor: Editor): DeskItem[] {
  const fromCards = listDeskItemsFromEditor(editor)
  const fromCharts: DeskItem[] = editor
    .getCurrentPageShapes()
    .filter(isBoardChartShape)
    .map((shape) => ({
      id: shape.id,
      type: "chart" as const,
      chart_type: shape.props.chart_type,
      title: shape.props.title,
      labels: Array.isArray(shape.props.labels) ? shape.props.labels.map(String) : [],
      values: Array.isArray(shape.props.values) ? shape.props.values.map((v) => Number(v) || 0) : [],
    }))
  return [...fromCards, ...fromCharts]
}

function findQuestionByPrompt(editor: Editor, prompt: string) {
  const needle = normalizePrompt(prompt)
  if (!needle) return null
  const questions = editor.getCurrentPageShapes().filter(isBoardQuestionShape)
  const exact = questions.find((shape) => normalizePrompt(shape.props.prompt) === needle)
  if (exact) return exact
  return questions.find((shape) => stemsAreSameProblem(shape.props.prompt, prompt)) ?? null
}

function asShapeId(id: string | undefined | null): TLShapeId | null {
  if (!id) return null
  return id as TLShapeId
}

function trackerCardTitle(source: string, cardTitle?: string): string {
  const header = (cardTitle || "").trim()
  if (!header) return source
  if (!source.trim() || /^typed problem$/i.test(source.trim())) return header
  return source
}

export function DeskProvider({ children }: { children: ReactNode }) {
  const { rememberQuestionStem } = useTaskTracker()
  const [deskItems, setDeskItems] = useState<DeskItem[]>([])
  const [drawer, setDrawer] = useState<Drawer>(null)
  const [activeProblem, setActiveProblem] = useState<ActiveProblem | null>(null)
  const editorRef = useRef<Editor | null>(null)
  const pendingOps = useRef<PendingOp[]>([])
  const pendingByQuestion = useRef(new Map<string, PendingOp[]>())
  const measureReady = useRef(new Set<string>())
  const applyCanvasOpRef = useRef<(op: CanvasOp, meta?: CanvasOpMeta) => void>(
    () => {},
  )
  const pendingReplace = useRef<DeskItem[] | null>(null)
  const unsubRef = useRef<(() => void) | null>(null)
  const lastQuestionId = useRef<TLShapeId | null>(null)
  const trackerShapeIds = useRef<Map<string, string>>(new Map())
  const pendingTrackerOps = useRef<Map<string, PendingOp[]>>(new Map())
  const clearedTrackerTurns = useRef<Set<string>>(new Set())
  const sessionsByRequest = useRef(new Map<string, CanvasSolveSession>())
  const sessionsByQuestion = useRef(new Map<string, CanvasSolveSession>())
  const solveHandlerRef = useRef<SolveHandler | null>(null)

  const syncFromEditor = useCallback((editor: Editor) => {
    setDeskItems(projectDeskItems(editor))
  }, [])

  const rememberSession = useCallback((session: CanvasSolveSession) => {
    sessionsByRequest.current.set(session.requestId, session)
    sessionsByQuestion.current.set(session.questionShapeId, session)
  }, [])

  const resolveAnchor = useCallback(
    (editor: Editor, anchorId?: string, requestId?: string): TLShapeId | null => {
      if (anchorId) {
        const shape = editor.getShape(asShapeId(anchorId)!)
        if (shape && isBoardQuestionShape(shape)) return shape.id
        if (shape && isBoardStepShape(shape)) {
          const parentId = asShapeId(shape.props.parentId)
          if (parentId && editor.getShape(parentId) && isBoardQuestionShape(editor.getShape(parentId)!)) {
            return parentId
          }
        }
      }
      if (requestId) {
        const session = sessionsByRequest.current.get(requestId)
        if (session && editor.getShape(session.questionShapeId)) return session.questionShapeId
      }
      if (lastQuestionId.current && editor.getShape(lastQuestionId.current)) {
        return lastQuestionId.current
      }
      const questions = editor.getCurrentPageShapes().filter(isBoardQuestionShape)
      return questions.length > 0 ? questions[questions.length - 1].id : null
    },
    [],
  )

  const ensureSession = useCallback(
    (editor: Editor, questionId: TLShapeId, requestId: string): CanvasSolveSession | null => {
      const existing =
        sessionsByQuestion.current.get(questionId) || sessionsByRequest.current.get(requestId)
      if (existing && existing.questionShapeId === questionId) {
        sessionsByRequest.current.set(requestId, existing)
        return existing
      }
      const session = beginSolveSession(editor, questionId, requestId)
      if (!session) return null
      rememberSession(session)
      setQuestionStatus(editor, questionId, "solving", requestId)
      return session
    },
    [rememberSession],
  )

  const spawnQuestion = useCallback((prompt: string, source = "", forceNew = false): string | null => {
    const trimmed = sanitizeCanvasPrompt(prompt)
    const pending = !isRenderableQuestionPrompt(trimmed)
    const editor = editorRef.current
    if (!editor) {
      pendingOps.current.push({ op: { op: "question", prompt: trimmed, source, freshCard: forceNew || undefined } })
      return null
    }
    if (!pending && !forceNew) {
      const existing = findQuestionByPrompt(editor, trimmed)
      if (existing) {
        lastQuestionId.current = existing.id
        return existing.id
      }
    }
    const id = spawnQuestionShape(editor, pending ? "" : trimmed, pending ? "" : source, { pending })
    lastQuestionId.current = id
    measureReady.current.delete(id)
    if (!pending) focusOnShape(editor, id)
    syncFromEditor(editor)
    return id
  }, [syncFromEditor])

  const getActiveQuestionId = useCallback((): string | null => {
    const editor = editorRef.current
    const id = lastQuestionId.current
    if (!editor || !id) return null
    const shape = editor.getShape(id)
    if (!shape || !isBoardQuestionShape(shape)) return null
    return id
  }, [])

  const focusTrackerQuestion = useCallback((trackerQuestionId: string) => {
    const editor = editorRef.current
    const remembered = trackerShapeIds.current.get(trackerQuestionId)
    const shapeId = remembered ? asShapeId(remembered) : null
    if (!editor || !shapeId || !editor.getShape(shapeId)) return
    focusOnShape(editor, shapeId)
  }, [])

  const readTrackerQuestionPrompt = useCallback((trackerQuestionId: string) => {
    const editor = editorRef.current
    const remembered = trackerShapeIds.current.get(trackerQuestionId)
    const shapeId = remembered ? asShapeId(remembered) : null
    if (!editor || !shapeId) return ""
    const shape = editor.getShape(shapeId)
    if (!shape || !isBoardQuestionShape(shape)) return ""
    return (shape.props.prompt ?? "").trim()
  }, [])

  const queueForQuestion = useCallback((questionId: string, op: CanvasOp, meta?: CanvasOpMeta) => {
    const list = pendingByQuestion.current.get(questionId) ?? []
    list.push({
      op,
      anchorId: meta?.canvasAnchorId,
      requestId: meta?.requestId,
      trackerQuestionId: meta?.trackerQuestionId,
    })
    pendingByQuestion.current.set(questionId, list)
  }, [])

  const questionReadyForChildren = useCallback((editor: Editor, questionId: TLShapeId): boolean => {
    if (!measureReady.current.has(questionId)) return false
    const shape = editor.getShape(questionId)
    if (!shape || !isBoardQuestionShape(shape)) return false
    if ((shape.opacity ?? 1) < 1) return false
    return isRenderableQuestionPrompt(shape.props.prompt) && (shape.props.h ?? 0) > 8
  }, [])

  const notifyQuestionMeasured = useCallback((questionId: string) => {
    measureReady.current.add(questionId)
    const queued = pendingByQuestion.current.get(questionId)
    if (!queued?.length) return
    pendingByQuestion.current.delete(questionId)
    const rank = (item: PendingOp) => {
      if (item.op.op === "figure") return 0
      if (item.op.op === "step") return 1
      if (item.op.op === "result") return 2
      return 3
    }
    queued.sort((a, b) => rank(a) - rank(b))
    for (const item of queued) {
      applyCanvasOpRef.current(item.op, {
        canvasAnchorId: item.anchorId,
        requestId: item.requestId,
        trackerQuestionId: item.trackerQuestionId,
      })
    }
    const editor = editorRef.current
    const id = asShapeId(questionId)
    if (!editor || !id) return
    reflowMainColumn(editor, id)
    groupQuestionCluster(editor, id)
  }, [])
  questionMeasureGate.notify = notifyQuestionMeasured

  const applyCanvasOp = useCallback(
    (op: CanvasOp, incoming?: CanvasOpMeta) => {
      let meta = incoming
      const editor = editorRef.current
      if (!editor) {
        pendingOps.current.push({
          op,
          anchorId: meta?.canvasAnchorId,
          requestId: meta?.requestId,
          trackerQuestionId: meta?.trackerQuestionId,
        })
        return
      }

      if (op.target === "chat") return

      if (meta?.trackerQuestionId) {
        const remembered = trackerShapeIds.current.get(meta.trackerQuestionId)
        const shapeId = remembered ? asShapeId(remembered) : null
        const shape = shapeId ? editor.getShape(shapeId) : null
        if (shapeId && shape && isBoardQuestionShape(shape)) {
          if (!meta.canvasAnchorId) meta = { ...meta, canvasAnchorId: shape.id }
          const turnKey = meta.requestId
          if (turnKey && !clearedTrackerTurns.current.has(turnKey)) {
            clearSolution(editor, shape.id)
            sessionsByQuestion.current.delete(shape.id)
            for (const [requestKey, session] of sessionsByRequest.current) {
              if (session.questionShapeId === shape.id) sessionsByRequest.current.delete(requestKey)
            }
            clearedTrackerTurns.current.add(turnKey)
            lastQuestionId.current = shape.id
          }
        } else if (remembered) {
          trackerShapeIds.current.delete(meta.trackerQuestionId)
        }
      }

      const bindTrackerShape = () => {
        const trackerQuestionId = meta?.trackerQuestionId
        if (trackerQuestionId && lastQuestionId.current) {
          const anchor = lastQuestionId.current
          trackerShapeIds.current.set(trackerQuestionId, anchor)
          const shape = editor.getShape(anchor)
          if (shape && isBoardQuestionShape(shape)) {
            const stored = (shape.props.prompt ?? "").trim()
            if (isSubstantialProblemStem(stored)) rememberQuestionStem(trackerQuestionId, stored)
          }
          const queued = pendingTrackerOps.current.get(trackerQuestionId)
          if (queued?.length) {
            pendingTrackerOps.current.delete(trackerQuestionId)
            for (const item of queued) {
              applyCanvasOpRef.current(item.op, {
                canvasAnchorId: anchor,
                requestId: item.requestId,
                trackerQuestionId,
              })
            }
          }
        }
      }
      const freshTrackerCard = Boolean(
        meta?.trackerQuestionId && !trackerShapeIds.current.get(meta.trackerQuestionId),
      )

      if (op.op === "question") {
        const prompt = isolateProblemStem(op.prompt)
        const source = sanitizeCanvasPrompt(op.source ?? "")
        const incomingSubstantial = isSubstantialProblemStem(prompt)
        const incomingPlaceholder = isPlaceholderQuestionPrompt(prompt)
        const incomingCoach = !prompt || isCoachSpeakPrompt(prompt)
        const titledSource = trackerCardTitle(source, meta?.cardTitle)
        if (freshTrackerCard) {
          if (!(incomingCoach && !incomingSubstantial) && !(incomingPlaceholder && !incomingSubstantial)) {
            spawnQuestion(prompt, titledSource)
            bindTrackerShape()
          }
          return
        }
        if (op.freshCard) {
          if (!(incomingCoach && !incomingSubstantial) && !(incomingPlaceholder && !incomingSubstantial)) {
            const currentId = lastQuestionId.current
            const current = currentId ? editor.getShape(currentId) : null
            if (current && isBoardQuestionShape(current)) {
              setQuestionStatus(editor, current.id, "solved")
            }
            spawnQuestion(prompt, titledSource, true)
            bindTrackerShape()
          }
          return
        }
        const existingId =
          resolveAnchor(editor, meta?.canvasAnchorId, meta?.requestId) || lastQuestionId.current
        const existing = existingId ? editor.getShape(existingId) : null

        // Coach-speak and chat summaries must never write the parent card.
        if (incomingCoach && !incomingSubstantial) {
          if (existing && isBoardQuestionShape(existing)) {
            lastQuestionId.current = existing.id
          }
          bindTrackerShape()
          return
        }

        if (incomingPlaceholder && !incomingSubstantial) {
          if (existing && isBoardQuestionShape(existing)) {
            lastQuestionId.current = existing.id
          }
          bindTrackerShape()
          return
        }

        if (existing && isBoardQuestionShape(existing)) {
          const currentPrompt = existing.props.prompt ?? ""
          const bookNavParent = isBookNavigationPrompt(currentPrompt)
          const currentSubstantial =
            isSubstantialProblemStem(currentPrompt) && !bookNavParent
          const currentPlaceholder =
            isPlaceholderQuestionPrompt(currentPrompt) ||
            bookNavParent ||
            (existing.opacity ?? 1) < 1 ||
            (existing.props.h ?? 0) <= 8
          const sameProblem = stemsAreSameProblem(currentPrompt, prompt)
          // Placeholder / book-nav cards hydrate from an isolated stem.
          // A substantial card hydrates in place for OCR/spelling jitter,
          // and only spawns when the incoming stem is a clearly different problem.
          const needsHydrate =
            (!currentSubstantial && currentPlaceholder && incomingSubstantial) ||
            (bookNavParent && incomingSubstantial)
          if (currentSubstantial && incomingSubstantial && !sameProblem && !bookNavParent) {
            setQuestionStatus(editor, existing.id, "solved")
            spawnQuestion(prompt, titledSource)
            bindTrackerShape()
            return
          }
          if (!needsHydrate && !(sameProblem && incomingSubstantial)) {
            lastQuestionId.current = existing.id
            syncFromEditor(editor)
            bindTrackerShape()
            return
          }
          editor.updateShape({
            id: existing.id,
            type: BOARD_QUESTION_TYPE,
            opacity: 1,
            props: {
              prompt,
              source: titledSource || existing.props.source,
              title: titledSource || existing.props.title,
              w: QUESTION_W,
              h: QUESTION_H,
            },
          })
          lastQuestionId.current = existing.id
          measureReady.current.delete(existing.id)
          groupQuestionCluster(editor, existing.id)
          focusOnShape(editor, existing.id)
          syncFromEditor(editor)
          bindTrackerShape()
          return
        }
        spawnQuestion(prompt, titledSource)
        bindTrackerShape()
        return
      }

      const requestId = meta?.requestId || "legacy"
      const rawAnchor = meta?.canvasAnchorId ? editor.getShape(asShapeId(meta.canvasAnchorId)!) : null
      const lateralFromId =
        (op.op === "step" && "branchFromId" in op && typeof op.branchFromId === "string" && op.branchFromId) ||
        (rawAnchor && isBoardStepShape(rawAnchor) ? rawAnchor.id : "")
      const anchor = resolveAnchor(editor, meta?.canvasAnchorId, meta?.requestId)
      const childOp =
        op.op === "step" || op.op === "result" || op.op === "chart" || op.op === "diagram" || op.op === "figure"

      if (freshTrackerCard && meta?.trackerQuestionId && childOp) {
        const list = pendingTrackerOps.current.get(meta.trackerQuestionId) ?? []
        list.push({
          op,
          anchorId: meta.canvasAnchorId,
          requestId: meta.requestId,
          trackerQuestionId: meta.trackerQuestionId,
        })
        pendingTrackerOps.current.set(meta.trackerQuestionId, list)
        return
      }

      if (!anchor) {
        const fallback = lastQuestionId.current
        if (fallback && childOp) {
          queueForQuestion(fallback, op, meta)
          return
        }
        if (op.op === "chart") {
          spawnChartNode(editor, null, { x: 72, y: 96 }, op)
          syncFromEditor(editor)
        } else if (op.op === "diagram") {
          spawnDiagramPrimitives(editor, op, { x: 72, y: 96 })
          syncFromEditor(editor)
        } else if (op.op === "figure") {
          spawnFigureNode(editor, null, op.image_url, op.index ?? 0)
          syncFromEditor(editor)
        }
        return
      }

      if (childOp && !questionReadyForChildren(editor, anchor)) {
        queueForQuestion(anchor, op, meta)
        return
      }

      const session = ensureSession(editor, anchor, requestId)
      if (session) syncSessionToQuestion(editor, session)
      if (op.op === "step") {
        const replaceLast =
          Boolean("replaceLast" in op && op.replaceLast) && !lateralFromId
        if (replaceLast && session) {
          const updated = updateLastMainColumnStep(editor, session, op.latex)
          if (!updated) spawnStepNode(editor, session, op.latex, op.index, lateralFromId, requestId)
        } else if (session) {
          spawnStepNode(editor, session, op.latex, op.index, lateralFromId, requestId)
        }
      } else if (lateralFromId) {
        // Why/How laterals are step-only — never also spawn RESULT/chart/diagram
        // into the main solution column.
        syncFromEditor(editor)
        return
      } else if (op.op === "result") {
        if (session) spawnResultNode(editor, session, formatBoardValue(op.value), op.summary || "")
      } else if (op.op === "chart") {
        spawnChartNode(editor, session, { x: 72, y: 96 }, op)
      } else if (op.op === "diagram") {
        if (session?.hasDiagram) {
          syncFromEditor(editor)
          return
        }
        const bounds = editor.getShapePageBounds(anchor)
        const origin = session?.cursor ?? {
          x: bounds ? bounds.maxX + 40 : 72,
          y: bounds?.y ?? 96,
        }
        spawnDiagramPrimitives(editor, op, origin)
        if (session) {
          session.hasDiagram = true
          session.cursor.x = origin.x
          session.cursor.y = origin.y + 320
        }
      } else if (op.op === "figure") {
        spawnFigureNode(editor, session, op.image_url, op.index ?? 0)
      }
      syncFromEditor(editor)
    },
    [ensureSession, queueForQuestion, questionReadyForChildren, rememberQuestionStem, resolveAnchor, spawnQuestion, syncFromEditor],
  )
  applyCanvasOpRef.current = applyCanvasOp

  const flushPending = useCallback(
    (editor: Editor) => {
      if (pendingReplace.current) {
        const boardTypes = new Set<string>([
          BOARD_QUESTION_TYPE,
          BOARD_STEP_TYPE,
          BOARD_RESULT_TYPE,
          BOARD_CHART_TYPE,
          BOARD_FIGURE_TYPE,
          BOARD_BRANCH_TYPE,
        ])
        const drop = editor
          .getCurrentPageShapes()
          .filter((shape) => isDeskCardShape(shape) || boardTypes.has(shape.type))
        editor.deleteShapes(drop.map((shape) => shape.id))
        spawnLegacyDeskItems(editor, pendingReplace.current)
        pendingReplace.current = null
        pendingOps.current = []
        syncFromEditor(editor)
        return
      }
      if (pendingOps.current.length > 0) {
        const queued = pendingOps.current
        pendingOps.current = []
        for (const item of queued) {
          applyCanvasOp(item.op, {
            canvasAnchorId: item.anchorId,
            requestId: item.requestId,
            trackerQuestionId: item.trackerQuestionId,
          })
        }
      }
    },
    [applyCanvasOp, syncFromEditor],
  )

  const setEditor = useCallback(
    (editor: Editor | null) => {
      unsubRef.current?.()
      unsubRef.current = null
      editorRef.current = editor
      if (!editor) return
      migrateDeskCards(editor)
      flushPending(editor)
      syncFromEditor(editor)
      unsubRef.current = editor.store.listen(
        () => {
          if (editorRef.current === editor) syncFromEditor(editor)
        },
        { scope: "document" },
      )
    },
    [flushPending, syncFromEditor],
  )

  const pushDeskUpdate = useCallback(
    (update: Record<string, unknown>, meta?: CanvasOpMeta): string | null => {
      if (isCanvasOp(update)) {
        applyCanvasOp(update, meta)
        return lastQuestionId.current
      }
      if (!isDeskUpdate(update)) return null
      const ops = deskUpdateToCanvasOps(update)
      for (const op of ops) applyCanvasOp(op, meta)
      return lastQuestionId.current
    },
    [applyCanvasOp],
  )

  const requestSolve = useCallback(
    (shapeId: string, opts?: { mode?: SolveMode }) => {
      const editor = editorRef.current
      const id = asShapeId(shapeId)
      if (!editor || !id) return
      const shape = editor.getShape(id)
      if (!shape || !isBoardQuestionShape(shape)) return
      if (shape.props.status === "solving") return
      clearSolution(editor, id)
      const requestId = crypto.randomUUID()
      const session = beginSolveSession(editor, id, requestId)
      if (!session) return
      rememberSession(session)
      setQuestionStatus(editor, id, "solving", requestId)
      lastQuestionId.current = id
      solveHandlerRef.current?.(shape.props.prompt, id, requestId, opts?.mode ?? "detailed")
    },
    [rememberSession],
  )

  const completeSolve = useCallback(
    (requestId?: string, shapeId?: string, status?: "idle" | "solved" | "error") => {
      const editor = editorRef.current
      const session =
        (requestId ? sessionsByRequest.current.get(requestId) : undefined) ||
        (shapeId ? sessionsByQuestion.current.get(shapeId) : undefined)
      const questionId = session?.questionShapeId || asShapeId(shapeId)
      if (!editor || !questionId) return
      const shape = editor.getShape(questionId)
      const childCount =
        shape && isBoardQuestionShape(shape) ? asShapeIdList(shape.props.childIds).length : 0
      const hasChildren = Boolean(session && session.stepShapeIds.length > 0) || childCount > 0
      if (session && session.status === "solving") {
        session.status = status === "error" ? "error" : "complete"
      }
      let next: "idle" | "solved" | "error"
      if (status === "error") {
        next = "error"
      } else if (hasChildren || session?.hasResult) {
        next = "solved"
      } else {
        next = status ?? "idle"
      }
      setQuestionStatus(editor, questionId, next, requestId)
      reflowMainColumn(editor, questionId)
      if (next === "solved") {
        applyClusterVisibility(editor, questionId)
        groupQuestionCluster(editor, questionId)
      }
    },
    [],
  )

  const toggleSolution = useCallback((shapeId: string) => {
    const editor = editorRef.current
    const id = asShapeId(shapeId)
    if (!editor || !id) return
    const shape = editor.getShape(id)
    if (!shape || !isBoardQuestionShape(shape)) return
    const hasChildren = asShapeIdList(shape.props.childIds).length > 0
    if (shape.props.status === "solving" && !hasChildren) return
    if (shape.props.status === "solved" || hasChildren) {
      setSolutionHidden(editor, id, !shape.props.solutionHidden)
    } else {
      requestSolve(shapeId)
    }
  }, [requestSolve])

  const registerSolveHandler = useCallback((handler: SolveHandler | null) => {
    solveHandlerRef.current = handler
  }, [])

  const removeDeskItem = useCallback((id: string) => {
    const editor = editorRef.current
    if (editor) {
      const shape = editor.getShape(deskCardShapeId(id)) ?? editor.getShape(id as TLShapeId)
      if (shape) editor.deleteShapes([shape.id])
      syncFromEditor(editor)
    }
  }, [syncFromEditor])

  const patchDeskItem = useCallback((id: string, patch: Partial<DeskItemMeta>) => {
    const editor = editorRef.current
    if (!editor) return
    const shape = editor.getShape(deskCardShapeId(id))
    if (!shape || !isDeskCardShape(shape)) return
    const current = readDeskItem(shape.props.item)
    if (!current) return
    editor.updateShape({
      id: shape.id,
      type: shape.type,
      props: { item: serializeDeskItem({ ...current, ...patch }) },
    })
    syncFromEditor(editor)
  }, [syncFromEditor])

  const stampHeadersOnItems = useCallback((ids: string[], title: string, description: string) => {
    if (!title.trim()) return
    const editor = editorRef.current
    const questionId = lastQuestionId.current
    if (!editor || !questionId) return
    const shape = editor.getShape(questionId)
    if (!shape || !isBoardQuestionShape(shape)) return
    editor.updateShape({
      id: shape.id,
      type: BOARD_QUESTION_TYPE,
      props: {
        source: title,
        title: title,
        // Never copy the AI summary / routing leftover into the canvas body.
      },
    })
    void ids
    void description
  }, [])

  const clearDeskItems = useCallback(() => {
    pendingOps.current = []
    pendingReplace.current = []
    sessionsByRequest.current.clear()
    sessionsByQuestion.current.clear()
    lastQuestionId.current = null
    pendingByQuestion.current.clear()
    measureReady.current.clear()
    const editor = editorRef.current
    if (!editor) {
      setDeskItems([])
      return
    }
    const boardTypes = new Set<string>([
      BOARD_QUESTION_TYPE,
      BOARD_STEP_TYPE,
      BOARD_RESULT_TYPE,
      BOARD_CHART_TYPE,
      BOARD_FIGURE_TYPE,
    ])
    const drop = editor
      .getCurrentPageShapes()
      .filter((shape) => isDeskCardShape(shape) || boardTypes.has(shape.type))
    editor.deleteShapes(drop.map((shape) => shape.id))
    syncFromEditor(editor)
  }, [syncFromEditor])

  const resetCanvasQueue = useCallback(() => {
    pendingOps.current = []
    pendingReplace.current = null
    pendingByQuestion.current.clear()
    measureReady.current.clear()
  }, [])

  const replaceDeskItems = useCallback((items: DeskItem[]) => {
    pendingOps.current = []
    pendingByQuestion.current.clear()
    measureReady.current.clear()
    const editor = editorRef.current
    if (editor) {
      pendingReplace.current = null
      const boardTypes = new Set<string>([
        BOARD_QUESTION_TYPE,
        BOARD_STEP_TYPE,
        BOARD_RESULT_TYPE,
        BOARD_CHART_TYPE,
        BOARD_FIGURE_TYPE,
      ])
      const drop = editor
        .getCurrentPageShapes()
        .filter((shape) => isDeskCardShape(shape) || boardTypes.has(shape.type))
      editor.deleteShapes(drop.map((shape) => shape.id))
      spawnLegacyDeskItems(editor, items)
      syncFromEditor(editor)
      return
    }
    pendingReplace.current = items
    setDeskItems(items)
  }, [syncFromEditor])

  const toggleDrawer = useCallback((id: Exclude<Drawer, null>) => {
    setDrawer((prev) => (prev === id ? null : id))
  }, [])

  const closeDrawer = useCallback(() => setDrawer(null), [])

  const value = useMemo(
    () => ({
      deskItems,
      pushDeskUpdate,
      applyCanvasOp,
      spawnQuestion,
      getActiveQuestionId,
      focusTrackerQuestion,
      readTrackerQuestionPrompt,
      notifyQuestionMeasured,
      requestSolve,
      toggleSolution,
      completeSolve,
      registerSolveHandler,
      removeDeskItem,
      patchDeskItem,
      stampHeadersOnItems,
      clearDeskItems,
      replaceDeskItems,
      resetCanvasQueue,
      setEditor,
      drawer,
      toggleDrawer,
      closeDrawer,
      activeProblem,
      setActiveProblem,
    }),
    [
      deskItems,
      pushDeskUpdate,
      applyCanvasOp,
      spawnQuestion,
      getActiveQuestionId,
      focusTrackerQuestion,
      readTrackerQuestionPrompt,
      notifyQuestionMeasured,
      requestSolve,
      toggleSolution,
      completeSolve,
      registerSolveHandler,
      removeDeskItem,
      patchDeskItem,
      stampHeadersOnItems,
      clearDeskItems,
      replaceDeskItems,
      resetCanvasQueue,
      setEditor,
      drawer,
      toggleDrawer,
      closeDrawer,
      activeProblem,
    ],
  )

  return <DeskContext.Provider value={value}>{children}</DeskContext.Provider>
}

export function useDesk() {
  return useContext(DeskContext)
}
