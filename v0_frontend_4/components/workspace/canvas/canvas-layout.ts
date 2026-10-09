import { createShapeId, type Editor, type TLShape, type TLShapeId } from "tldraw"
import { isRenderableQuestionPrompt, sanitizeCanvasPrompt } from "@/lib/parse-task-assignment"
import {
  BOARD_BRANCH_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_QUESTION_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_STEP_TYPE,
  COLLAPSED_CHILD,
  asShapeIdList,
  isBoardBranchShape,
  isBoardChartShape,
  isBoardFigureShape,
  isBoardQuestionShape,
  isBoardResultShape,
  isBoardStepShape,
  isCollapsedBoardShape,
  stepExplanationHidden,
  type BoardBranchShape,
  type BoardQuestionShape,
  type BoardQuestionStatus,
  type BoardStepShape,
  type ExpandedGeometry,
} from "./board-constants"

export const STEP_GAP = 20
/** Main column is left-aligned with the question — no flowchart indent. */
export const STEP_INDENT = 0
export const BRANCH_GAP = 28
export const COLUMN_WIDTH = 740
export const QUESTION_W = 680
export const QUESTION_H = 148
export const COMPACT_QUESTION_H = 44
export const COMPACT_QUESTION_W = 380
/** Main solution column — steps, result. */
export const STEP_W = 680
/** Lateral Why/How side-notes stay narrower than the main column. */
export const LATERAL_STEP_W = 540
export const STEP_H = 96
export const BRANCH_PAD = 12
export const BRANCH_HEADER_H = 22
export const RESULT_H = 96
export const CHART_W = 640
export const CHART_H = 268
export const FIGURE_W = 680
export const FIGURE_H = 160
export const QUESTION_ORIGIN = { x: 72, y: 96 }
export const QUESTION_GUTTER = 80
export const QUESTION_WRAP_Y = 360
export const QUESTION_COLUMNS = 3
export const QUESTION_IDLE_MS = 180_000
/** Max simultaneous explanation blocks per lateral branch container — a new
 * Why/How request on the same anchor replaces the old one instead of the
 * side column growing without bound. */
export const LATERAL_BLOCK_LIMIT = 2

export type CanvasSolveSession = {
  requestId: string
  questionShapeId: TLShapeId
  boardId: string
  cursor: { x: number; y: number }
  prevShapeId: TLShapeId
  stepShapeIds: TLShapeId[]
  seenSteps: Set<number>
  /** Offset added to inbound main-column indices when a new request reuses
   * this session so Socratic follow-up hints append instead of colliding
   * with `seenSteps` / `index: 0`. */
  stepIndexBase: number
  hasResult: boolean
  hasChart: boolean
  hasDiagram: boolean
  status: "solving" | "complete" | "error"
}

export function listQuestionShapes(editor: Editor): BoardQuestionShape[] {
  return editor.getCurrentPageShapes().filter(isBoardQuestionShape)
}

/** Bottom edge of a question's full solved cluster (question + steps/result/
 * chart/branches), so row-wrapping never lands a new question inside a
 * taller-than-expected previous cluster. */
function clusterBottomY(editor: Editor, question: BoardQuestionShape): number {
  const bounds = editor.getShapePageBounds(question.id)
  let maxY = bounds ? bounds.maxY : question.y + QUESTION_H
  for (const childId of asShapeIdList(question.props.childIds)) {
    const childBounds = editor.getShapePageBounds(childId)
    if (childBounds) maxY = Math.max(maxY, childBounds.maxY)
  }
  return maxY
}

export function nextQuestionPoint(editor: Editor): { x: number; y: number } {
  const questions = listQuestionShapes(editor)
  const n = questions.length
  const col = n % QUESTION_COLUMNS
  const row = Math.floor(n / QUESTION_COLUMNS)
  const x = QUESTION_ORIGIN.x + col * (QUESTION_W + QUESTION_GUTTER)
  if (row === 0) {
    return { x, y: QUESTION_ORIGIN.y }
  }
  // Wrap below the tallest cluster in the row above (not a fixed row height)
  // so a question whose steps/branches grew past QUESTION_WRAP_Y never gets
  // a sibling spawned on top of it.
  const rowAbove = questions.slice((row - 1) * QUESTION_COLUMNS, row * QUESTION_COLUMNS)
  let maxBottom = QUESTION_ORIGIN.y + (row - 1) * QUESTION_WRAP_Y
  for (const q of rowAbove) {
    maxBottom = Math.max(maxBottom, clusterBottomY(editor, q))
  }
  return { x, y: maxBottom + QUESTION_GUTTER }
}

const FOCUS_PADDING = 80
const FOCUS_ANIMATION_MS = 320

function boundsContains(
  outer: { x: number; y: number; w: number; h: number },
  inner: { x: number; y: number; w: number; h: number },
): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.w <= outer.x + outer.w &&
    inner.y + inner.h <= outer.y + outer.h
  )
}

/** Pan (or, if it wouldn't fit at the current zoom, fit) the camera so
 * `shapeId` is brought into view. A no-op if it is already fully visible,
 * so this never fights the user's own camera during an active session. */
export function focusOnShape(editor: Editor, shapeId: TLShapeId) {
  const bounds = editor.getShapePageBounds(shapeId)
  if (!bounds) return
  const target = { x: bounds.x, y: bounds.y, w: bounds.w, h: bounds.h }
  const viewport = editor.getViewportPageBounds()
  const viewportBox = { x: viewport.x, y: viewport.y, w: viewport.w, h: viewport.h }
  const padded = {
    x: target.x - FOCUS_PADDING,
    y: target.y - FOCUS_PADDING,
    w: target.w + FOCUS_PADDING * 2,
    h: target.h + FOCUS_PADDING * 2,
  }
  if (boundsContains(viewportBox, padded)) return

  const fitsAtCurrentZoom = padded.w <= viewportBox.w && padded.h <= viewportBox.h
  if (fitsAtCurrentZoom) {
    editor.centerOnPoint(
      { x: target.x + target.w / 2, y: target.y + target.h / 2 },
      { animation: { duration: FOCUS_ANIMATION_MS } },
    )
  } else {
    editor.zoomToBounds(target, {
      inset: FOCUS_PADDING,
      animation: { duration: FOCUS_ANIMATION_MS },
    })
  }
}

function boxesOverlap(
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y
}

export function collisionShift(
  editor: Editor,
  x: number,
  y: number,
  w: number,
  h: number,
  ignore: Iterable<string>,
): { x: number; y: number } {
  const skip = new Set(ignore)
  let nx = x
  for (let attempt = 0; attempt < 8; attempt++) {
    const box = { x: nx, y, w, h }
    const hit = editor.getCurrentPageShapes().some((shape) => {
      if (skip.has(shape.id)) return false
      const bounds = editor.getShapePageBounds(shape.id)
      if (!bounds) return false
      return boxesOverlap(box, { x: bounds.x, y: bounds.y, w: bounds.w, h: bounds.h })
    })
    if (!hit) return { x: nx, y }
    nx += COLUMN_WIDTH
  }
  return { x: nx, y }
}

/** Move a shape so its page origin sits at (pageX, pageY), whether it lives
 * on the page or inside a tldraw group / branch container. */
function localPointForPage(
  editor: Editor,
  shape: TLShape,
  pageX: number,
  pageY: number,
): { x: number; y: number } {
  const pageT = editor.getShapePageTransform(shape.id)
  const origin = pageT ? pageT.applyToPoint({ x: 0, y: 0 }) : { x: shape.x, y: shape.y }
  return { x: shape.x + (pageX - origin.x), y: shape.y + (pageY - origin.y) }
}

function setShapePagePosition(editor: Editor, id: TLShapeId, pageX: number, pageY: number) {
  const shape = editor.getShape(id)
  if (!shape) return
  const next = localPointForPage(editor, shape, pageX, pageY)
  if (Math.abs(next.x - shape.x) < 0.5 && Math.abs(next.y - shape.y) < 0.5) return
  if (shape.isLocked) {
    editor.updateShape({ id, type: shape.type, isLocked: false })
  }
  editor.updateShape({
    id,
    type: shape.type,
    x: next.x,
    y: next.y,
  })
  if (shape.isLocked) {
    editor.updateShape({ id, type: shape.type, isLocked: true })
  }
}

function shapesParentedTo(editor: Editor, parentId: TLShapeId): TLShape[] {
  return editor.getCurrentPageShapes().filter((shape) => shape.parentId === parentId)
}

function isParkableBoardShape(shape: TLShape) {
  return (
    isBoardStepShape(shape) ||
    isBoardResultShape(shape) ||
    isBoardChartShape(shape) ||
    isBoardFigureShape(shape) ||
    isBoardBranchShape(shape)
  )
}

function parkableType(
  shape: TLShape,
): "board-step" | "board-result" | "board-chart" | "board-figure" | "board-branch" {
  return shape.type as "board-step" | "board-result" | "board-chart" | "board-figure" | "board-branch"
}

function findBranchContainer(
  editor: Editor,
  question: BoardQuestionShape,
  branchFromId: string,
): BoardBranchShape | null {
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (child && isBoardBranchShape(child) && child.props.branchFromId === branchFromId) {
      return child
    }
  }
  return null
}

/** Bind a native tldraw arrow. Used only for lateral Why/How branches
 * (right edge of the parent step → left edge of the explanation module).
 * Main-column steps never call this. */
export function bindArrow(
  editor: Editor,
  fromId: TLShapeId,
  toId: TLShapeId,
  side: "vertical" | "lateral" = "lateral",
): TLShapeId | null {
  const fromBounds = editor.getShapePageBounds(fromId)
  const toBounds = editor.getShapePageBounds(toId)
  if (!fromBounds || !toBounds) return null

  const arrowId = createShapeId()
  const start =
    side === "lateral"
      ? { x: fromBounds.maxX, y: fromBounds.y + fromBounds.h / 2 }
      : { x: fromBounds.x + fromBounds.w / 2, y: fromBounds.maxY }
  const end =
    side === "lateral"
      ? { x: toBounds.x, y: toBounds.y + toBounds.h / 2 }
      : { x: toBounds.x + toBounds.w / 2, y: toBounds.y }

  try {
    editor.createShape({
      id: arrowId,
      type: "arrow",
      x: start.x,
      y: start.y,
      props: {
        start: { x: 0, y: 0 },
        end: { x: end.x - start.x, y: end.y - start.y },
        arrowheadStart: "none",
        arrowheadEnd: "arrow",
      },
    })
  } catch {
    return null
  }

  const bind = (terminal: "start" | "end", shapeId: TLShapeId, anchor: { x: number; y: number }) => {
    try {
      editor.createBinding({
        type: "arrow",
        fromId: arrowId,
        toId: shapeId,
        props: {
          terminal,
          normalizedAnchor: anchor,
          isExact: false,
          isPrecise: true,
          snap: "none",
        },
      })
    } catch {
      /* unbound arrow still points at the node */
    }
  }

  if (side === "lateral") {
    bind("start", fromId, { x: 1, y: 0.5 })
    bind("end", toId, { x: 0, y: 0.5 })
  } else {
    bind("start", fromId, { x: 0.5, y: 1 })
    bind("end", toId, { x: 0.5, y: 0 })
  }
  return arrowId
}

export function beginSolveSession(
  editor: Editor,
  questionId: TLShapeId,
  requestId: string,
  boardId = "",
): CanvasSolveSession | null {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return null
  const bounds = editor.getShapePageBounds(questionId)
  if (!bounds) return null
  return {
    requestId,
    questionShapeId: questionId,
    boardId,
    cursor: { x: bounds.x, y: bounds.maxY + STEP_GAP },
    prevShapeId: questionId,
    stepShapeIds: [],
    seenSteps: new Set(),
    stepIndexBase: 0,
    hasResult: false,
    hasChart: false,
    hasDiagram: false,
    status: "solving",
  }
}

/**
 * Refresh the session's cursor/anchor from the question's *current* page
 * bounds before every spawn. Without this, dragging the question mid-stream
 * leaves later steps at stale coordinates because the session snapshotted
 * its cursor once at solve-start.
 */
export function syncSessionToQuestion(editor: Editor, session: CanvasSolveSession) {
  const qBounds = editor.getShapePageBounds(session.questionShapeId)
  if (!qBounds) return
  if (session.stepShapeIds.length === 0) {
    session.cursor = { x: qBounds.x, y: qBounds.maxY + STEP_GAP }
    return
  }
  let maxY = qBounds.maxY
  for (const id of session.stepShapeIds) {
    const shape = editor.getShape(id)
    if (!shape || isBoardStepShape(shape) && shape.props.branchFromId) continue
    const bounds = editor.getShapePageBounds(id)
    if (bounds) maxY = Math.max(maxY, bounds.maxY)
  }
  session.cursor = { x: qBounds.x, y: maxY + STEP_GAP }
}

export function setQuestionStatus(
  editor: Editor,
  questionId: TLShapeId,
  status: BoardQuestionStatus,
  requestId?: string,
) {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return
  editor.updateShape({
    id: shape.id,
    type: BOARD_QUESTION_TYPE,
    props: {
      status,
      requestId: requestId ?? shape.props.requestId,
    },
  })
}

function rememberChildren(
  editor: Editor,
  questionId: TLShapeId,
  childId: TLShapeId,
  arrowId: TLShapeId | null,
) {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return
  const childIds = [...asShapeIdList(shape.props.childIds), childId]
  const arrowIds = arrowId
    ? [...asShapeIdList(shape.props.arrowIds), arrowId]
    : asShapeIdList(shape.props.arrowIds)
  editor.updateShape({
    id: shape.id,
    type: BOARD_QUESTION_TYPE,
    props: { childIds, arrowIds },
  })
}

export function clearSolution(editor: Editor, questionId: TLShapeId) {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return
  const drop = [...asShapeIdList(shape.props.childIds), ...asShapeIdList(shape.props.arrowIds)].filter(
    (id) => Boolean(editor.getShape(id)),
  )
  if (drop.length > 0) editor.deleteShapes(drop)
  editor.updateShape({
    id: shape.id,
    type: BOARD_QUESTION_TYPE,
    props: {
      childIds: [],
      arrowIds: [],
      status: "idle",
      requestId: "",
      solutionHidden: false,
    },
  })
}

function placeLinkedNode(
  editor: Editor,
  session: CanvasSolveSession,
  id: TLShapeId,
  indent: number,
  w: number,
  h: number,
) {
  const ignore = new Set<string>([session.questionShapeId, ...session.stepShapeIds])
  const shifted = collisionShift(editor, session.cursor.x, session.cursor.y, w + indent, h, ignore)
  editor.updateShape({
    id,
    type: editor.getShape(id)?.type ?? BOARD_STEP_TYPE,
    x: shifted.x + indent,
    y: shifted.y,
  })
  // Main column: stacked cards only. Never draw tldraw arrows between steps.
  rememberChildren(editor, session.questionShapeId, id, null)
  session.cursor = { x: shifted.x, y: shifted.y + h + STEP_GAP }
  session.prevShapeId = id
  session.stepShapeIds.push(id)
}

/** Clears a stale explanation branch container the moment a *new* Why/How
 * request starts on the same anchor step, so the new explanation's steps
 * land in a clean container instead of appending alongside (or only
 * partially overwriting) the previous request's blocks. No-op once this
 * request has already been primed (idempotent across the whole stream). */
function primeBranchContainerForRequest(
  editor: Editor,
  session: CanvasSolveSession,
  branchFromId: string,
  requestId: string,
) {
  const question = editor.getShape(session.questionShapeId)
  if (!question || !isBoardQuestionShape(question)) return
  const container = findBranchContainer(editor, question, branchFromId)
  if (!container) return
  if ((container.props.lastRequestId || "") === requestId) return
  const stale = shapesParentedTo(editor, container.id).map((child) => child.id)
  if (stale.length > 0) editor.deleteShapes(stale)
  editor.updateShape({
    id: container.id,
    type: BOARD_BRANCH_TYPE,
    props: { lastRequestId: requestId },
  })
}

function nextMainColumnStepIndex(editor: Editor, questionId: TLShapeId): number {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return 0
  let max = -1
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (child && isBoardStepShape(child) && !child.props.branchFromId) {
      max = Math.max(max, Number(child.props.index) || 0)
    }
  }
  return max + 1
}

/** Overwrite the latest main-column step in place. Does not bump stepIndexBase. */
export function updateLastMainColumnStep(
  editor: Editor,
  session: CanvasSolveSession,
  latex: string,
): TLShapeId | null {
  const question = editor.getShape(session.questionShapeId)
  if (!question || !isBoardQuestionShape(question)) return null
  let last: BoardStepShape | null = null
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (!child || !isBoardStepShape(child) || child.props.branchFromId) continue
    if (!last || Number(child.props.index) >= Number(last.props.index)) {
      last = child
    }
  }
  if (!last) return null
  if (last.props.latex !== latex) {
    editor.updateShape({ id: last.id, type: BOARD_STEP_TYPE, props: { latex } })
  }
  reflowMainColumn(editor, session.questionShapeId)
  return last.id
}

export function spawnStepNode(
  editor: Editor,
  session: CanvasSolveSession,
  latex: string,
  index: number,
  branchFromId = "",
  requestId?: string,
): TLShapeId | null {
  const isLateral = Boolean(branchFromId)
  // `session` may be a long-lived object reused across many turns on the
  // same question, so its own `.requestId` can be stale — prefer the actual
  // per-turn id passed in by the caller when deciding if this is a new
  // explanation request.
  const turnRequestId = requestId || session.requestId
  if (!isLateral && session.requestId !== turnRequestId) {
    session.requestId = turnRequestId
    session.stepIndexBase = nextMainColumnStepIndex(editor, session.questionShapeId)
    session.seenSteps = new Set()
  }
  const resolvedIndex = isLateral ? index : (session.stepIndexBase || 0) + index
  if (!branchFromId && session.seenSteps.has(index)) return null
  if (isLateral) {
    primeBranchContainerForRequest(editor, session, branchFromId, turnRequestId)
  }
  const question = editor.getShape(session.questionShapeId)
  const existing = question && isBoardQuestionShape(question)
    ? findExistingStep(editor, question, resolvedIndex, branchFromId)
    : null
  if (existing) {
    if (existing.props.latex !== latex) {
      editor.updateShape({ id: existing.id, type: BOARD_STEP_TYPE, props: { latex } })
    }
    if (!isLateral) session.seenSteps.add(index)
    return existing.id
  }
  if (isLateral) {
    const containerId = ensureBranchContainer(editor, session, branchFromId, turnRequestId)
    const blockCount = shapesParentedTo(editor, containerId).filter(isBoardStepShape).length
    if (blockCount >= LATERAL_BLOCK_LIMIT) return null
  }
  const w = isLateral ? LATERAL_STEP_W : STEP_W
  const id = createShapeId()
  editor.createShape({
    id,
    type: BOARD_STEP_TYPE,
    x: session.cursor.x + STEP_INDENT,
    y: session.cursor.y,
    props: {
      w,
      h: STEP_H,
      latex,
      index: resolvedIndex,
      parentId: session.questionShapeId,
      branchFromId,
      isExplanationBranch: isLateral,
      explanationHidden: false,
    },
  })
  if (!isLateral) session.seenSteps.add(index)
  if (isLateral) {
    placeBranchNode(editor, session, id, branchFromId, w, STEP_H, turnRequestId)
  } else {
    placeLinkedNode(editor, session, id, STEP_INDENT, w, STEP_H)
    reflowMainColumn(editor, session.questionShapeId)
  }
  return id
}

function findExistingStep(
  editor: Editor,
  question: BoardQuestionShape,
  index: number,
  branchFromId: string,
) {
  const candidates: TLShape[] = []
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (!child) continue
    if (isBoardStepShape(child)) candidates.push(child)
    if (isBoardBranchShape(child) && child.props.branchFromId === (branchFromId || child.props.branchFromId)) {
      for (const nested of shapesParentedTo(editor, child.id)) {
        if (isBoardStepShape(nested)) candidates.push(nested)
      }
    }
  }
  for (const child of candidates) {
    if (!isBoardStepShape(child)) continue
    if (child.props.index !== index) continue
    if ((child.props.branchFromId || "") !== (branchFromId || "")) continue
    return child
  }
  return null
}

function ensureBranchContainer(
  editor: Editor,
  session: CanvasSolveSession,
  branchFromId: string,
  requestId: string,
): TLShapeId {
  const question = editor.getShape(session.questionShapeId)
  if (question && isBoardQuestionShape(question)) {
    const existing = findBranchContainer(editor, question, branchFromId)
    if (existing) return existing.id
  }
  const parentBounds = editor.getShapePageBounds(branchFromId as TLShapeId)
  const origin = parentBounds
    ? { x: parentBounds.maxX + BRANCH_GAP, y: parentBounds.y }
    : session.cursor
  const id = createShapeId()
  editor.createShape({
    id,
    type: BOARD_BRANCH_TYPE,
    x: origin.x,
    y: origin.y,
    props: {
      w: LATERAL_STEP_W + BRANCH_PAD * 2,
      h: STEP_H + BRANCH_HEADER_H + BRANCH_PAD * 2,
      parentId: session.questionShapeId,
      branchFromId,
      lastRequestId: requestId,
    },
  })
  return id
}

/** Place a lateral Why/How child inside the branch container to the right of its anchor. */
function placeBranchNode(
  editor: Editor,
  session: CanvasSolveSession,
  id: TLShapeId,
  branchFromId: string,
  w: number,
  h: number,
  requestId: string,
) {
  const containerId = ensureBranchContainer(editor, session, branchFromId, requestId)
  const question = editor.getShape(session.questionShapeId)
  let isNewContainer = false
  if (question && isBoardQuestionShape(question)) {
    isNewContainer = !asShapeIdList(question.props.childIds).includes(containerId)
  }

  editor.reparentShapes([id], containerId)
  editor.updateShape({
    id,
    type: BOARD_STEP_TYPE,
    x: BRANCH_PAD,
    y: BRANCH_PAD + BRANCH_HEADER_H,
    props: { w, h },
  })

  const arrowId = isNewContainer
    ? bindArrow(editor, branchFromId as TLShapeId, containerId, "lateral")
    : null
  if (isNewContainer) {
    rememberChildren(editor, session.questionShapeId, containerId, arrowId)
    groupQuestionCluster(editor, session.questionShapeId)
  }
  session.stepShapeIds.push(id)
  reflowBranchContainer(editor, containerId)

  const parent = editor.getShape(branchFromId as TLShapeId)
  if (parent && isBoardStepShape(parent) && stepExplanationHidden(parent)) {
    setBranchExplanationHidden(editor, branchFromId as TLShapeId, true)
  }
}

export function reflowLateralBranchFromStep(editor: Editor, stepId: TLShapeId) {
  const step = editor.getShape(stepId)
  if (!step || !isBoardStepShape(step)) return
  const anchorId = (step.props.branchFromId || step.id) as TLShapeId
  const questionId = step.props.parentId as TLShapeId
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return
  const container = findBranchContainer(editor, question, anchorId)
  if (container) reflowBranchContainer(editor, container.id)
}

export function reflowVisibleLateralBranches(editor: Editor, questionId: TLShapeId) {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (!child || !isBoardBranchShape(child)) continue
    if (isCollapsedBoardShape(child.props)) continue
    const anchor = editor.getShape(child.props.branchFromId as TLShapeId)
    if (anchor && isBoardStepShape(anchor) && stepExplanationHidden(anchor)) continue
    reflowBranchContainer(editor, child.id)
  }
}

function reflowBranchContainer(editor: Editor, containerId: TLShapeId) {
  const container = editor.getShape(containerId)
  if (!container || !isBoardBranchShape(container)) return
  if (isCollapsedBoardShape(container.props)) return
  const anchor = editor.getShape(container.props.branchFromId as TLShapeId)
  const parentBounds = anchor ? editor.getShapePageBounds(anchor.id) : null
  if (parentBounds) {
    setShapePagePosition(editor, containerId, parentBounds.maxX + BRANCH_GAP, parentBounds.y)
  }

  const nested = shapesParentedTo(editor, containerId)
    .filter(isBoardStepShape)
    .filter((child) => !isCollapsedBoardShape(child.props))
    .sort((a, b) => a.props.index - b.props.index || a.y - b.y)

  let cursorY = BRANCH_PAD + BRANCH_HEADER_H
  let innerW = LATERAL_STEP_W
  for (const child of nested) {
    const h = Math.max(Number(child.props.h) || STEP_H, 1)
    innerW = Math.max(innerW, Number(child.props.w) || LATERAL_STEP_W)
    editor.updateShape({
      id: child.id,
      type: BOARD_STEP_TYPE,
      x: BRANCH_PAD,
      y: cursorY,
      opacity: 1,
    })
    cursorY += h + STEP_GAP
  }
  const nextH = nested.length === 0
    ? BRANCH_HEADER_H + BRANCH_PAD * 2 + STEP_H
    : cursorY - STEP_GAP + BRANCH_PAD
  const nextW = innerW + BRANCH_PAD * 2
  if (Math.abs(nextW - container.props.w) > 1 || Math.abs(nextH - container.props.h) > 1) {
    editor.updateShape({
      id: container.id,
      type: BOARD_BRANCH_TYPE,
      opacity: 1,
      props: { w: nextW, h: nextH },
    })
  } else {
    editor.updateShape({
      id: container.id,
      type: BOARD_BRANCH_TYPE,
      opacity: 1,
    })
  }
}

export function spawnResultNode(
  editor: Editor,
  session: CanvasSolveSession,
  value: string,
  summary: string,
): TLShapeId | null {
  if (session.hasResult) return null
  const id = createShapeId()
  editor.createShape({
    id,
    type: BOARD_RESULT_TYPE,
    x: session.cursor.x,
    y: session.cursor.y,
    props: {
      w: STEP_W,
      h: RESULT_H,
      value,
      summary,
      parentId: session.questionShapeId,
    },
  })
  session.hasResult = true
  placeLinkedNode(editor, session, id, 0, STEP_W, RESULT_H)
  reflowMainColumn(editor, session.questionShapeId)
  groupQuestionCluster(editor, session.questionShapeId)
  setQuestionStatus(editor, session.questionShapeId, "solved", session.requestId)
  session.status = "complete"
  return id
}

export function spawnChartNode(
  editor: Editor,
  session: CanvasSolveSession | null,
  origin: { x: number; y: number },
  chart: {
    chart_type: "line" | "bar" | "pie"
    title: string
    labels: string[]
    values: number[]
  },
): TLShapeId | null {
  if (session?.hasChart) return null
  const cursor = session?.cursor ?? origin
  const ignore = session
    ? new Set<string>([session.questionShapeId, ...session.stepShapeIds])
    : new Set<string>()
  const shifted = collisionShift(editor, cursor.x, cursor.y, CHART_W, CHART_H, ignore)
  const id = createShapeId()
  editor.createShape({
    id,
    type: BOARD_CHART_TYPE,
    x: shifted.x,
    y: shifted.y,
    props: {
      w: CHART_W,
      h: CHART_H,
      chart_type: chart.chart_type,
      title: chart.title,
      labels: chart.labels,
      values: chart.values,
      parentId: session?.questionShapeId ?? "",
    },
  })
  if (session) {
    session.hasChart = true
    rememberChildren(editor, session.questionShapeId, id, null)
    session.cursor = { x: shifted.x, y: shifted.y + CHART_H + STEP_GAP }
    session.prevShapeId = id
    session.stepShapeIds.push(id)
  }
  return id
}

export function spawnFigureNode(
  editor: Editor,
  session: CanvasSolveSession | null,
  imageUrl: string,
  index = 0,
): TLShapeId | null {
  const questionId = session?.questionShapeId
  if (questionId) {
    const question = editor.getShape(questionId)
    if (question && isBoardQuestionShape(question)) {
      for (const childId of asShapeIdList(question.props.childIds)) {
        const child = editor.getShape(childId)
        if (child && isBoardFigureShape(child) && child.props.imageUrl === imageUrl) {
          return child.id
        }
      }
    }
  }
  const id = createShapeId()
  const qBounds = questionId ? editor.getShapePageBounds(questionId) : null
  editor.createShape({
    id,
    type: BOARD_FIGURE_TYPE,
    x: qBounds?.x ?? session?.cursor.x ?? 72,
    y: qBounds?.y ?? session?.cursor.y ?? 96,
    props: {
      // Parked 1×1 data holder — the question card paints the crop inline.
      w: questionId ? COLLAPSED_CHILD : FIGURE_W,
      h: questionId ? COLLAPSED_CHILD : FIGURE_H,
      imageUrl,
      index,
      parentId: questionId ?? "",
    },
  })
  if (session && questionId) {
    rememberChildren(editor, questionId, id, null)
    session.stepShapeIds.push(id)
    groupQuestionCluster(editor, questionId)
    reflowMainColumn(editor, questionId)
  }
  return id
}

export function spawnQuestionShape(
  editor: Editor,
  prompt: string,
  source = "",
  opts?: { title?: string; origin?: "freeform" | "tracker"; pending?: boolean },
): TLShapeId {
  const point = nextQuestionPoint(editor)
  const id = createShapeId()
  const cleanPrompt = sanitizeCanvasPrompt(prompt)
  const cleanTitle = sanitizeCanvasPrompt(opts?.title ?? "")
  const cleanSource = sanitizeCanvasPrompt(source)
  const pending = Boolean(opts?.pending) || !isRenderableQuestionPrompt(cleanPrompt)
  editor.createShape({
    id,
    type: BOARD_QUESTION_TYPE,
    x: point.x,
    y: point.y,
    opacity: pending ? 0 : 1,
    props: {
      w: QUESTION_W,
      // h: 1 keeps the placeholder out of the column; w stays QUESTION_W so
      // this is not treated as a parked/collapsed 1×1 child.
      h: pending ? 1 : QUESTION_H,
      title: pending ? "" : cleanTitle || cleanSource,
      prompt: pending ? "" : cleanPrompt,
      source: pending ? "" : cleanSource,
      status: "idle",
      requestId: "",
      childIds: [],
      arrowIds: [],
      origin: opts?.origin ?? "freeform",
      solutionHidden: (opts?.origin ?? "freeform") === "tracker",
      isPromptHidden: false,
    },
  })
  return id
}

// ---------------------------------------------------------------------------
// Store-level collapse / expand (Show/Hide cascade)
// ---------------------------------------------------------------------------

function questionPageBox(editor: Editor, questionId: TLShapeId) {
  return editor.getShapePageBounds(questionId)
}

function clusterChildIds(question: BoardQuestionShape): TLShapeId[] {
  return [...asShapeIdList(question.props.childIds), ...asShapeIdList(question.props.arrowIds)]
}

/** Every board-step in the cluster, at any lateral nesting depth. Branch
 * containers — however deep the logical Why/How chain — are always parented
 * directly to the question, so one flat pass over `childIds` (plus each
 * container's own direct step children) already reaches every nested step. */
function collectClusterStepIds(editor: Editor, question: BoardQuestionShape): TLShapeId[] {
  const ids: TLShapeId[] = []
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (!child) continue
    if (isBoardStepShape(child)) ids.push(child.id)
    if (isBoardBranchShape(child)) {
      for (const nested of shapesParentedTo(editor, child.id)) {
        if (isBoardStepShape(nested)) ids.push(nested.id)
      }
    }
  }
  return ids
}

function questionPageOrigin(editor: Editor, question: BoardQuestionShape): { x: number; y: number } {
  const bounds = editor.getShapePageBounds(question.id)
  return { x: bounds?.x ?? question.x, y: bounds?.y ?? question.y }
}

function childrenLockedHidden(question: BoardQuestionShape): boolean {
  return Boolean(question.props.isPromptHidden || question.props.solutionHidden)
}

function patchCollapsedBox(
  editor: Editor,
  child: TLShape & { props: { w: number; h: number; expandedGeometry?: ExpandedGeometry } },
  park: { x: number; y: number },
) {
  // Never skip already-1x1 nodes — always snap x/y to the park origin.
  const alreadyCollapsed = isCollapsedBoardShape(child.props)
  const stored = child.props.expandedGeometry
  const storedGood = Boolean(stored) && Number(stored?.w) > COLLAPSED_CHILD && Number(stored?.h) > COLLAPSED_CHILD

  let expandedGeometry: ExpandedGeometry | undefined = stored
  if (!alreadyCollapsed) {
    // Only snapshot while expanded. Never overwrite a good cached geometry
    // with a 1x1 box from a nested / second park pass.
    expandedGeometry = { x: child.x, y: child.y, w: Number(child.props.w), h: Number(child.props.h) }
  } else if (!storedGood) {
    expandedGeometry = stored
  }

  const type = parkableType(child)
  editor.updateShape({ id: child.id, type, isLocked: false })
  editor.updateShape({
    id: child.id,
    type,
    x: park.x,
    y: park.y,
    opacity: 0,
    isLocked: false,
    props: { w: COLLAPSED_CHILD, h: COLLAPSED_CHILD, expandedGeometry },
  })
  editor.updateShape({ id: child.id, type, isLocked: true })
}

/** Zero out step/result/chart/branch store bounds and park arrows, so the
 * tldraw group bounding box actually shrinks. Already-1x1 children are still
 * snapped to `parkPage` (page space) — never skipped. Nested laterals use
 * local (0,0) inside their container. */
function collapseBoardShapes(
  editor: Editor,
  ids: TLShapeId[],
  parkPage: { x: number; y: number },
  local = false,
) {
  for (const id of ids) {
    const child = editor.getShape(id)
    if (!child) continue
    if (child.type === "arrow") {
      const park = local ? parkPage : localPointForPage(editor, child, parkPage.x, parkPage.y)
      editor.updateShape({ id: child.id, type: "arrow", isLocked: false })
      editor.updateShape({
        id: child.id,
        type: "arrow",
        x: park.x,
        y: park.y,
        opacity: 0,
        isLocked: false,
      })
      editor.updateShape({ id: child.id, type: "arrow", isLocked: true })
      continue
    }
    if (!isParkableBoardShape(child)) continue
    if (isBoardBranchShape(child)) {
      collapseBoardShapes(
        editor,
        shapesParentedTo(editor, child.id).map((nested) => nested.id),
        { x: 0, y: 0 },
        true,
      )
    }
    const fresh = editor.getShape(id) ?? child
    const localPark = local ? parkPage : localPointForPage(editor, fresh, parkPage.x, parkPage.y)
    patchCollapsedBox(
      editor,
      fresh as TLShape & { props: { w: number; h: number; expandedGeometry?: ExpandedGeometry } },
      localPark,
    )
  }
}

function expandBoardShapes(editor: Editor, ids: TLShapeId[]) {
  for (const id of ids) {
    const child = editor.getShape(id)
    if (!child) continue
    if (child.type === "arrow") {
      editor.updateShape({ id: child.id, type: "arrow", opacity: 1, isLocked: false })
      continue
    }
    if (!isParkableBoardShape(child)) continue
    if (isBoardFigureShape(child)) continue
    const props = child.props as { expandedGeometry?: ExpandedGeometry }
    const geo = props.expandedGeometry
    if (!geo || geo.w <= COLLAPSED_CHILD || geo.h <= COLLAPSED_CHILD) continue
    editor.updateShape({
      id: child.id,
      type: parkableType(child),
      opacity: 1,
      isLocked: false,
      props: { w: geo.w, h: geo.h },
    })
    if (isBoardBranchShape(child)) {
      expandBoardShapes(
        editor,
        shapesParentedTo(editor, child.id).map((nested) => nested.id),
      )
    }
  }
}

function mainColumnRank(shape: TLShape | undefined): number {
  if (!shape) return 9
  if (isBoardFigureShape(shape)) return 0
  if (isBoardStepShape(shape)) return 1
  if (isBoardResultShape(shape)) return 2
  if (isBoardChartShape(shape)) return 3
  return 4
}

/** Re-stack main-column children (steps, then result/chart) top to bottom
 * under the question. Figures sit inside the question card, not below it.
 * Lateral Why/How branches and collapsed (parked) children are left alone.
 * Vertical flowchart arrows are stripped. */
export function reflowMainColumn(editor: Editor, questionId: TLShapeId) {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return
  if ((question.opacity ?? 1) < 1) return
  if (!isRenderableQuestionPrompt(question.props.prompt)) return
  if ((question.props.h ?? 0) <= 8) return
  const qBounds = questionPageBox(editor, questionId)
  if (!qBounds) return

  const mainIds = asShapeIdList(question.props.childIds).filter((id) => {
    const shape = editor.getShape(id)
    if (!shape) return false
    if (isCollapsedBoardShape(shape.props as { w: number; h: number })) return false
    if (isBoardBranchShape(shape)) return false
    if (isBoardStepShape(shape)) return !shape.props.branchFromId
    // Figures render inside the question card (flex-row), not in the stack.
    return isBoardResultShape(shape) || isBoardChartShape(shape)
  })
  const mainSet = new Set<string>([questionId, ...mainIds])
  const staleArrows = asShapeIdList(question.props.arrowIds).filter((arrowId) => {
    const arrow = editor.getShape(arrowId)
    if (!arrow || arrow.type !== "arrow") return false
    const bindings = editor.getBindingsInvolvingShape(arrow.id)
    const ends = bindings.map((b) => b.toId)
    return ends.length > 0 && ends.every((id) => mainSet.has(id))
  })
  if (staleArrows.length > 0) {
    editor.deleteShapes(staleArrows)
    editor.updateShape({
      id: question.id,
      type: BOARD_QUESTION_TYPE,
      props: {
        arrowIds: asShapeIdList(question.props.arrowIds).filter((id) => !staleArrows.includes(id)),
      },
    })
  }

  mainIds.sort((a, b) => {
    const sa = editor.getShape(a)
    const sb = editor.getShape(b)
    const ra = mainColumnRank(sa)
    const rb = mainColumnRank(sb)
    if (ra !== rb) return ra - rb
    if (sa && isBoardStepShape(sa) && sb && isBoardStepShape(sb)) {
      return sa.props.index - sb.props.index
    }
    const ba = editor.getShapePageBounds(a)
    const bb = editor.getShapePageBounds(b)
    return (ba?.y ?? 0) - (bb?.y ?? 0)
  })

  let cursorY = qBounds.y + Number(question.props.h) + STEP_GAP
  const x = qBounds.x + STEP_INDENT
  for (const id of mainIds) {
    const shape = editor.getShape(id)
    if (!shape) continue
    const h = Math.max(Number((shape.props as { h?: number }).h) || STEP_H, 1)
    const targetX = isBoardResultShape(shape) ? qBounds.x : x
    setShapePagePosition(editor, id, targetX, cursorY)
    cursorY += h + STEP_GAP
  }
  reflowVisibleLateralBranches(editor, questionId)
}

function applyQuestionVisibility(editor: Editor, questionId: TLShapeId, ids: TLShapeId[]) {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question) || !childrenLockedHidden(question)) return
  collapseBoardShapes(editor, ids, questionPageOrigin(editor, question))
}

/** Single cascade every visibility toggle routes through: Tier 1 (question),
 * Tier 2 (solution), and Tier 3 (per-step explanation) all write real store
 * bounds here instead of duplicating opacity logic three times. */
export function applyClusterVisibility(editor: Editor, questionId: TLShapeId) {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return
  const ids = clusterChildIds(question)
  const park = questionPageOrigin(editor, question)

  if (childrenLockedHidden(question)) {
    collapseBoardShapes(editor, ids, park)
    if (question.props.isPromptHidden) {
      editor.updateShape({ id: question.id, type: BOARD_QUESTION_TYPE, props: { h: COMPACT_QUESTION_H } })
    }
    return
  }

  expandBoardShapes(editor, ids)
  // Tier 3 wins over the freshly-expanded Tier 1/2: re-collapse laterals
  // whose own step still has explanationHidden set. Walk every step in the
  // cluster — including ones nested inside another explanation branch — so
  // a hidden sub-explanation stays hidden after a cascading re-expand.
  for (const stepId of collectClusterStepIds(editor, question)) {
    const child = editor.getShape(stepId)
    if (child && isBoardStepShape(child) && stepExplanationHidden(child)) {
      parkLateralBranch(editor, stepId)
    }
  }
  reflowMainColumn(editor, questionId)
  reflowVisibleLateralBranches(editor, questionId)
}

export function setQuestionPromptHidden(editor: Editor, questionId: TLShapeId, hidden: boolean) {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return
  if (Boolean(shape.props.isPromptHidden) === hidden) return
  editor.updateShape({
    id: shape.id,
    type: BOARD_QUESTION_TYPE,
    props: hidden
      ? { isPromptHidden: true, h: COMPACT_QUESTION_H }
      : { isPromptHidden: false },
  })
  applyClusterVisibility(editor, questionId)
}

export function setSolutionHidden(editor: Editor, questionId: TLShapeId, hidden: boolean) {
  const shape = editor.getShape(questionId)
  if (!shape || !isBoardQuestionShape(shape)) return
  editor.updateShape({ id: shape.id, type: BOARD_QUESTION_TYPE, props: { solutionHidden: hidden } })
  applyClusterVisibility(editor, questionId)
}

export function listLateralBranchIds(editor: Editor, stepId: TLShapeId): TLShapeId[] {
  const step = editor.getShape(stepId)
  if (!step || !isBoardStepShape(step)) return []
  const parent = editor.getShape(step.props.parentId as TLShapeId)
  if (!parent || !isBoardQuestionShape(parent)) return []
  const ids: TLShapeId[] = []
  const container = findBranchContainer(editor, parent, stepId)
  if (container) {
    ids.push(container.id)
    for (const nested of shapesParentedTo(editor, container.id)) ids.push(nested.id)
  }
  for (const childId of asShapeIdList(parent.props.childIds)) {
    const child = editor.getShape(childId)
    if (child && isBoardStepShape(child) && child.props.branchFromId === stepId) {
      ids.push(childId)
    }
  }
  const related = new Set(ids)
  related.add(stepId)
  if (container) related.add(container.id)
  for (const arrowId of asShapeIdList(parent.props.arrowIds)) {
    const arrow = editor.getShape(arrowId)
    if (!arrow) continue
    const bindings = editor.getBindingsInvolvingShape(arrow.id)
    if (bindings.some((b) => related.has(b.toId) || related.has(b.fromId))) ids.push(arrowId)
  }
  return [...new Set(ids)]
}

export function hasLateralBranch(editor: Editor, stepId: TLShapeId): boolean {
  return listLateralBranchIds(editor, stepId).some((id) => {
    const shape = editor.getShape(id)
    return Boolean(shape && (isBoardStepShape(shape) || isBoardBranchShape(shape)))
  })
}

function parkLateralBranch(editor: Editor, stepId: TLShapeId) {
  const step = editor.getShape(stepId)
  if (!step || !isBoardStepShape(step)) return
  const question = editor.getShape(step.props.parentId as TLShapeId)
  if (!question || !isBoardQuestionShape(question)) {
    const stepBounds = editor.getShapePageBounds(stepId)
    collapseBoardShapes(editor, listLateralBranchIds(editor, stepId), {
      x: stepBounds?.x ?? step.x,
      y: stepBounds?.y ?? step.y,
    })
    return
  }
  const parkPage = questionPageOrigin(editor, question)
  const container = findBranchContainer(editor, question, stepId)
  if (container) {
    // Cascade first: any of this container's own step children may anchor
    // a further nested explanation branch (a Why/How about this Why/How).
    // Collapsing the parent block must take that entire chain down with it.
    for (const nested of shapesParentedTo(editor, container.id)) {
      if (!isBoardStepShape(nested)) continue
      if (findBranchContainer(editor, question, nested.id)) {
        parkLateralBranch(editor, nested.id)
      }
    }
    collapseBoardShapes(
      editor,
      shapesParentedTo(editor, container.id).map((child) => child.id),
      { x: 0, y: 0 },
      true,
    )
    collapseBoardShapes(editor, [container.id], parkPage)
    const arrows = listLateralBranchIds(editor, stepId).filter((id) => editor.getShape(id)?.type === "arrow")
    collapseBoardShapes(editor, arrows, parkPage)
    return
  }
  collapseBoardShapes(editor, listLateralBranchIds(editor, stepId), parkPage)
}

export function setBranchExplanationHidden(editor: Editor, stepId: TLShapeId, hidden: boolean) {
  const step = editor.getShape(stepId)
  if (!step || !isBoardStepShape(step)) return
  editor.updateShape({ id: step.id, type: BOARD_STEP_TYPE, props: { explanationHidden: hidden } })

  const parent = editor.getShape(step.props.parentId as TLShapeId)
  if (parent && isBoardQuestionShape(parent)) {
    applyClusterVisibility(editor, parent.id)
    return
  }

  const ids = listLateralBranchIds(editor, stepId)
  if (hidden) {
    const stepBounds = editor.getShapePageBounds(stepId)
    collapseBoardShapes(editor, ids, { x: stepBounds?.x ?? step.x, y: stepBounds?.y ?? step.y })
  } else {
    expandBoardShapes(editor, ids)
    reflowLateralBranchFromStep(editor, stepId)
  }
}

export function toggleBranchExplanation(editor: Editor, stepId: TLShapeId) {
  const step = editor.getShape(stepId)
  if (!step || !isBoardStepShape(step)) return
  setBranchExplanationHidden(editor, stepId, !stepExplanationHidden(step))
}

function mainColumnGroupIds(editor: Editor, question: BoardQuestionShape): TLShapeId[] {
  const ids: TLShapeId[] = [question.id]
  for (const childId of asShapeIdList(question.props.childIds)) {
    const child = editor.getShape(childId)
    if (!child) continue
    if (isBoardStepShape(child) && (child.props.branchFromId || child.props.isExplanationBranch)) continue
    if (
      isBoardBranchShape(child) ||
      isBoardStepShape(child) ||
      isBoardResultShape(child) ||
      isBoardChartShape(child) ||
      isBoardFigureShape(child)
    ) {
      ids.push(child.id)
    }
  }
  for (const arrowId of asShapeIdList(question.props.arrowIds)) {
    const arrow = editor.getShape(arrowId)
    if (arrow && arrow.type === "arrow") ids.push(arrow.id)
  }
  return ids.filter((id) => Boolean(editor.getShape(id)))
}

function reparentOrphanLateralSteps(editor: Editor, question: BoardQuestionShape) {
  for (const childId of asShapeIdList(question.props.childIds)) {
    const container = editor.getShape(childId)
    if (!container || !isBoardBranchShape(container)) continue
    const fromId = container.props.branchFromId
    for (const shape of editor.getCurrentPageShapes()) {
      if (!isBoardStepShape(shape)) continue
      if (shape.parentId === container.id) continue
      const from = shape.props.branchFromId || ""
      if (from !== fromId && from !== container.id) continue
      editor.reparentShapes([shape.id], container.id)
    }
  }
}

export function groupQuestionCluster(editor: Editor, questionId: TLShapeId) {
  const question = editor.getShape(questionId)
  if (!question || !isBoardQuestionShape(question)) return
  const parent = editor.getShape(question.parentId)
  if (parent && parent.type === "group") {
    try {
      editor.ungroupShapes([parent.id])
    } catch {
      /* already ungrouped */
    }
  }
  const refreshed = editor.getShape(questionId)
  if (!refreshed || !isBoardQuestionShape(refreshed)) return
  reparentOrphanLateralSteps(editor, refreshed)
  const latest = editor.getShape(questionId)
  if (!latest || !isBoardQuestionShape(latest)) return
  const ids = mainColumnGroupIds(editor, latest)
  if (ids.length >= 2) {
    try {
      editor.groupShapes(ids)
    } catch {
      /* grouping is a convenience, not required for correctness */
    }
  }
  reflowVisibleLateralBranches(editor, questionId)
}
