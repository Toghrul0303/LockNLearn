"use client"

import { useEffect, useRef, type CSSProperties, type MouseEvent, type ReactNode } from "react"
import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
  resizeBox,
  useEditor,
  useValue,
  type Editor,
  type TLShape,
  type TLShapeId,
  type RecordProps,
} from "tldraw"
import { Bookmark } from "lucide-react"
import { cn } from "@/lib/utils"
import { isRenderableQuestionPrompt, sanitizeCanvasPrompt } from "@/lib/parse-task-assignment"
import { notifyQuestionPainted, useDesk } from "../desk-context"
import { useLanguage } from "../language-context"
import { useMemoryBox, type QuestionClusterShape } from "../memory-box-context"
import { BoardChartPlot } from "./board-chart-plot"
import { BoardRichText, formatBoardValue, onMathPointerDown } from "./board-math"
import {
  BOARD_BRANCH_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_QUESTION_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_STEP_TYPE,
  asShapeIdList,
  isBoardBranchShape,
  isBoardChartShape,
  isBoardFigureShape,
  isBoardQuestionShape,
  isBoardResultShape,
  isBoardStepShape,
  isCollapsedBoardShape,
  stepExplanationHidden,
} from "./board-constants"
import {
  QUESTION_H,
  groupQuestionCluster,
  hasLateralBranch,
  reflowLateralBranchFromStep,
  reflowMainColumn,
  setQuestionPromptHidden,
  toggleBranchExplanation,
} from "./canvas-layout"

export {
  BOARD_BRANCH_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_QUESTION_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_STEP_TYPE,
  isBoardBranchShape,
  isBoardChartShape,
  isBoardFigureShape,
  isBoardQuestionShape,
  isBoardResultShape,
  isBoardStepShape,
}

const COMPACT_ZOOM = 0.4
const QUESTION_IDLE_COLLAPSE_MS = 180_000
const REMEASURE_DELAYS_MS = [80, 240, 640]

const CARD_CLASS =
  "box-border flex w-full flex-col gap-2 rounded-xl border border-black/[0.08] bg-card px-3 py-4 shadow-[0_1px_0_rgba(0,0,0,0.03)]"

export type IBoardQuestionShape = TLShape<"board-question">
export type IBoardStepShape = TLShape<"board-step">
export type IBoardResultShape = TLShape<"board-result">
export type IBoardChartShape = TLShape<"board-chart">
export type IBoardFigureShape = TLShape<"board-figure">
export type IBoardBranchShape = TLShape<"board-branch">

function useCompact(shapeId: IBoardQuestionShape["id"] | IBoardStepShape["id"] | IBoardResultShape["id"] | IBoardChartShape["id"] | IBoardFigureShape["id"]) {
  const editor = useEditor()
  return useValue(
    "board-compact",
    () => {
      const zoom = editor.getZoomLevel()
      if (zoom < COMPACT_ZOOM) return true
      const pageBounds = editor.getShapePageBounds(shapeId)
      if (!pageBounds) return true
      return !editor.getViewportPageBounds().collides(pageBounds)
    },
    [editor, shapeId],
  )
}

/**
 * Measures the rendered content and pushes the result back into the tldraw
 * store as `props.h` so the box actually grows/shrinks with its content
 * (MathJax typesetting resolves asynchronously, so a single post-mount
 * measurement is not enough).
 */
function useMeasuredHeight(
  shape: {
    id: TLShapeId
    type: "board-question" | "board-step" | "board-result" | "board-chart"
    props: { w: number; h: number }
  },
  source: string,
  compact: boolean,
) {
  const editor = useEditor()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (compact) return
    const el = ref.current
    if (!el) return
    // A parked (collapsed) child must stay 1x1 — never let a late remeasure
    // re-inflate it after Show/Hide Solution or Show/Hide Explanation.
    if (isCollapsedBoardShape(shape.props)) return

    let cancelled = false

    const measure = () => {
      if (cancelled || compact) return
      const host = ref.current
      if (!host || !host.isConnected) return
      const zoom = editor.getZoomLevel()
      const pageBounds = editor.getShapePageBounds(shape.id)
      const viewport = editor.getViewportPageBounds()
      if (zoom < COMPACT_ZOOM || !pageBounds || !viewport.collides(pageBounds)) return
      const current = editor.getShape(shape.id)
      if (!current || isCollapsedBoardShape(current.props as { w: number; h: number })) return
      if (isBoardQuestionShape(current)) {
        if ((current.opacity ?? 1) < 1) return
        if (!isRenderableQuestionPrompt(current.props.prompt)) return
      }
      // Prefer scrollHeight so overflow:hidden frames cannot cap the reading
      // at the spawn default (RESULT_H / STEP_H) before MathJax finishes.
      const next = Math.max(1, Math.ceil(Math.max(host.offsetHeight, host.scrollHeight)))
      const prevH = (current.props as { h: number }).h
      if (isBoardQuestionShape(current)) {
        if (next <= 8) return
        if (Math.abs(next - prevH) > 4) {
          editor.updateShape({ id: shape.id, type: shape.type, props: { h: next } })
          reflowMainColumn(editor, current.id)
        }
        notifyQuestionPainted(current.id)
        return
      }
      if (Math.abs(next - prevH) > 4) {
        editor.updateShape({ id: shape.id, type: shape.type, props: { h: next } })
        if (isBoardStepShape(current) && current.props.branchFromId) {
          reflowLateralBranchFromStep(editor, current.id)
        } else {
          const parentId = (current.props as { parentId?: string }).parentId as TLShapeId | undefined
          if (parentId) reflowMainColumn(editor, parentId)
        }
      }
    }

    const timers = REMEASURE_DELAYS_MS.map((delay) => window.setTimeout(measure, delay))

    const resizeObserver = new ResizeObserver(() => measure())
    resizeObserver.observe(el)

    const mutationObserver = new MutationObserver(() => measure())
    mutationObserver.observe(el, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true,
    })

    return () => {
      cancelled = true
      timers.forEach((id) => window.clearTimeout(id))
      resizeObserver.disconnect()
      mutationObserver.disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor, shape.id, shape.type, source, compact])

  return ref
}

function BoardFrame({
  width,
  height,
  className,
  overflow = "hidden",
  children,
}: {
  width: number
  height: number
  className?: string
  overflow?: CSSProperties["overflow"]
  children: ReactNode
}) {
  return (
    <HTMLContainer
      className={cn("board-node", className)}
      style={{ width, height, background: "transparent", overflow }}
    >
      {children}
    </HTMLContainer>
  )
}

function toClusterShape(shape: TLShape): QuestionClusterShape {
  return {
    id: shape.id,
    type: shape.type,
    x: shape.x,
    y: shape.y,
    parentId: shape.parentId,
    props: JSON.parse(JSON.stringify(shape.props)) as Record<string, unknown>,
  }
}

/** Read-only copy of the question cluster. Does not update the store. */
function collectQuestionCluster(editor: Editor, questionId: TLShapeId): QuestionClusterShape[] {
  const root = editor.getShape(questionId)
  if (!root || !isBoardQuestionShape(root)) return []
  const seen = new Set<string>()
  const out: QuestionClusterShape[] = []
  const visit = (shape: TLShape) => {
    if (seen.has(shape.id)) return
    seen.add(shape.id)
    out.push(toClusterShape(shape))
    const props = shape.props as { childIds?: unknown; arrowIds?: unknown }
    for (const childId of asShapeIdList(props.childIds)) {
      const child = editor.getShape(childId)
      if (child) visit(child)
    }
    for (const arrowId of asShapeIdList(props.arrowIds)) {
      const arrow = editor.getShape(arrowId)
      if (arrow) visit(arrow)
    }
    if (isBoardBranchShape(shape)) {
      for (const child of editor.getCurrentPageShapes()) {
        if (child.parentId === shape.id) visit(child)
      }
    }
  }
  visit(root)
  return out
}

function QuestionBody({ shape }: { shape: IBoardQuestionShape }) {
  const editor = useEditor()
  const compact = useCompact(shape.id)
  const selected = useValue(
    "board-question-selected",
    () => editor.getSelectedShapeIds().includes(shape.id),
    [editor, shape.id],
  )
  const figures = useValue(
    "board-question-figures",
    () => {
      const current = editor.getShape(shape.id)
      if (!current || !isBoardQuestionShape(current)) return []
      return asShapeIdList(current.props.childIds)
        .map((id) => editor.getShape(id))
        .filter((child): child is IBoardFigureShape => Boolean(child && isBoardFigureShape(child)))
        .filter((child) => Boolean(child.props.imageUrl))
        .sort((a, b) => a.props.index - b.props.index)
    },
    [editor, shape.id],
  )
  const { requestSolve, toggleSolution } = useDesk()
  const { saveQuestionSnapshot } = useMemoryBox()
  const { t } = useLanguage()
  const renderable = isRenderableQuestionPrompt(shape.props.prompt)
  const figureKey = figures.map((figure) => figure.props.imageUrl).join("|")
  const hostRef = useMeasuredHeight(
    shape,
    `${shape.props.prompt}:${shape.props.status}:${shape.props.isPromptHidden}:${figureKey}`,
    compact,
  )
  const { status, isPromptHidden: promptHidden, solutionHidden, origin } = shape.props
  const childCount = asShapeIdList(shape.props.childIds).length
  const busy = status === "solving" && childCount === 0
  const showInlineFigures = figures.length > 0 && !promptHidden

  const onFigureLoad = () => {
    const current = editor.getShape(shape.id)
    if (!current || !isBoardQuestionShape(current)) return
    reflowMainColumn(editor, current.id)
    groupQuestionCluster(editor, current.id)
  }

  useEffect(() => {
    if (!renderable) return
    const el = hostRef.current
    if (!el) return
    let timer: number | undefined
    const reset = () => {
      if (timer) window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        const current = editor.getShape(shape.id)
        if (current && isBoardQuestionShape(current) && !current.props.isPromptHidden) {
          setQuestionPromptHidden(editor, shape.id, true)
        }
      }, QUESTION_IDLE_COLLAPSE_MS)
    }
    reset()
    el.addEventListener("pointerdown", reset)
    el.addEventListener("pointermove", reset)
    return () => {
      if (timer) window.clearTimeout(timer)
      el.removeEventListener("pointerdown", reset)
      el.removeEventListener("pointermove", reset)
    }
  }, [editor, hostRef, shape.id, renderable])

  const title = sanitizeCanvasPrompt(shape.props.title || shape.props.source) || t("desk.question")
  const body = sanitizeCanvasPrompt(shape.props.prompt)
  const showBody = Boolean(body) && body !== title && !promptHidden

  const onToggleCollapse = (event: MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    setQuestionPromptHidden(editor, shape.id, !promptHidden)
  }

  const onFlagClick = (event: MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    reflowMainColumn(editor, shape.id)
    saveQuestionSnapshot({
      questionId: shape.id,
      title,
      prompt: body,
      cluster: collectQuestionCluster(editor, shape.id),
    })
  }

  const onSolutionClick = (event: MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    if (busy) return
    if (status === "idle") requestSolve(shape.id)
    else toggleSolution(shape.id)
  }

  if (!renderable) {
    return (
      <BoardFrame width={shape.props.w} height={shape.props.h}>
        <div ref={hostRef} className="h-px w-full overflow-hidden" aria-hidden />
      </BoardFrame>
    )
  }

  if (compact) {
    return (
      <BoardFrame width={shape.props.w} height={shape.props.h}>
        <div ref={hostRef} className={CARD_CLASS}>
          <p className="truncate font-display text-sm font-semibold text-foreground/90">{title}</p>
        </div>
      </BoardFrame>
    )
  }

  const showSolutionButton = !promptHidden && (origin === "tracker" || status !== "idle" || childCount > 0)
  const solutionLabel = busy
    ? t("desk.solving")
    : status === "solved" || childCount > 0
      ? solutionHidden
        ? t("desk.showSolution")
        : t("desk.hideSolution")
      : t("desk.showSolution")

  return (
    <BoardFrame width={shape.props.w} height={shape.props.h} overflow="visible">
      <div ref={hostRef} className={CARD_CLASS}>
        <div className="flex items-start justify-between gap-2">
          <p
            data-highlight-source=""
            data-shape-id={shape.id}
            style={{ pointerEvents: "all", userSelect: "text" }}
            onPointerDown={onMathPointerDown}
            className="min-w-0 truncate text-[13px] font-medium text-muted-foreground"
          >
            {title}
          </p>
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              data-board-action=""
              aria-label={t("desk.flagQuestion")}
              title={t("desk.flagQuestion")}
              className="board-action"
              style={{ pointerEvents: "all" }}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onFlagClick}
            >
              <Bookmark className="size-3.5" aria-hidden="true" />
            </button>
            <button
              type="button"
              data-board-action=""
              className="board-action shrink-0"
              style={{ pointerEvents: "all" }}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onToggleCollapse}
            >
              {promptHidden ? t("desk.expandQuestion") : t("desk.collapseQuestion")}
            </button>
          </div>
        </div>

        {showBody || showInlineFigures ? (
          <div className={cn(showInlineFigures && "flex flex-row items-start gap-3")}>
            {showBody ? (
              <div
                data-highlight-source=""
                data-shape-id={shape.id}
                className={cn(
                  "min-w-0 text-[15px] leading-7 text-pretty text-slate-800",
                  showInlineFigures && "flex-1",
                )}
                onPointerDown={onMathPointerDown}
              >
                <BoardRichText text={body} />
              </div>
            ) : null}

            {showInlineFigures ? (
              <div className="w-[40%] max-w-[40%] shrink-0">
                {figures.map((figure) => (
                  <img
                    key={figure.id}
                    src={figure.props.imageUrl}
                    alt=""
                    draggable={false}
                    onLoad={onFigureLoad}
                    className="block w-full object-contain"
                  />
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {showSolutionButton ? (
          <div
            className={cn(
              "flex items-center gap-1 transition-opacity",
              selected || busy ? "opacity-100" : "opacity-60 hover:opacity-100",
            )}
          >
            <button
              type="button"
              data-board-action=""
              className="board-action"
              style={{ pointerEvents: busy ? "none" : "all" }}
              disabled={busy}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onSolutionClick}
            >
              {solutionLabel}
            </button>
          </div>
        ) : null}
      </div>
    </BoardFrame>
  )
}

function StepBody({ shape }: { shape: IBoardStepShape }) {
  const editor = useEditor()
  const compact = useCompact(shape.id)
  const hasLateral = useValue(
    "board-step-has-lateral",
    () => hasLateralBranch(editor, shape.id),
    [editor, shape.id],
  )
  const explanationHidden = stepExplanationHidden(shape)
  const bodyRef = useMeasuredHeight(shape, `${shape.props.latex}:${hasLateral}:${explanationHidden}`, compact)
  const { t } = useLanguage()

  const onToggleExplanation = (event: MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    toggleBranchExplanation(editor, shape.id)
  }

  if (compact) {
    return (
      <BoardFrame width={shape.props.w} height={shape.props.h}>
        <div className={CARD_CLASS}>
          <p className="truncate text-sm text-muted-foreground">{shape.props.index + 1}</p>
        </div>
      </BoardFrame>
    )
  }
  return (
    <BoardFrame width={shape.props.w} height={shape.props.h} overflow="visible">
      <div ref={bodyRef} className={CARD_CLASS}>
        <div
          data-highlight-source=""
          data-shape-id={shape.id}
          className="flex gap-2.5 text-sm leading-7 text-slate-800"
          onPointerDown={onMathPointerDown}
        >
          <span className="w-4 shrink-0 pt-0.5 text-[11px] font-medium tabular-nums text-muted-foreground/70">
            {shape.props.index + 1}
          </span>
          <BoardRichText text={shape.props.latex} block className="min-w-0 text-pretty text-slate-800 leading-7" />
        </div>
        {hasLateral ? (
          <div className="flex items-center gap-1 pl-6">
            <button
              type="button"
              data-board-action=""
              className="board-action"
              style={{ pointerEvents: "all" }}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onToggleExplanation}
            >
              {explanationHidden ? t("desk.showExplanation") : t("desk.hideExplanation")}
            </button>
          </div>
        ) : null}
      </div>
    </BoardFrame>
  )
}

function ResultBody({ shape }: { shape: IBoardResultShape }) {
  const compact = useCompact(shape.id)
  const { t } = useLanguage()
  const bodyRef = useMeasuredHeight(shape, `${shape.props.value}:${shape.props.summary}`, compact)
  if (compact) {
    return (
      <BoardFrame width={shape.props.w} height={shape.props.h} className="board-result">
        <p className="truncate px-1 font-display text-sm font-semibold">{t("desk.result")}</p>
      </BoardFrame>
    )
  }
  return (
    <BoardFrame width={shape.props.w} height={shape.props.h} className="board-result" overflow="visible">
      <div
        ref={bodyRef}
        data-highlight-source=""
        data-shape-id={shape.id}
        className={CARD_CLASS}
        onPointerDown={onMathPointerDown}
      >
        <span className="text-[10px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
          {t("desk.result")}
        </span>
        <BoardRichText
          text={formatBoardValue(shape.props.value)}
          className="text-xl font-semibold tracking-tight"
        />
        {shape.props.summary ? (
          <BoardRichText
            text={shape.props.summary}
            className="text-sm font-semibold leading-relaxed text-pretty text-slate-800"
          />
        ) : null}
      </div>
    </BoardFrame>
  )
}

function ChartBody({ shape }: { shape: IBoardChartShape }) {
  const compact = useCompact(shape.id)
  const chartType = shape.props.chart_type === "bar" || shape.props.chart_type === "pie"
    ? shape.props.chart_type
    : "line"
  if (compact) {
    return (
      <BoardFrame width={shape.props.w} height={shape.props.h}>
        <p className="truncate px-1 text-sm text-muted-foreground">{shape.props.title || "Chart"}</p>
      </BoardFrame>
    )
  }
  return (
    <BoardFrame width={shape.props.w} height={shape.props.h}>
      <div className="flex h-full flex-col gap-1 px-1 py-1">
        {shape.props.title ? (
          <p className="text-xs font-medium text-muted-foreground">{shape.props.title}</p>
        ) : null}
        <div className="min-h-0 flex-1">
          <BoardChartPlot chartType={chartType} labels={shape.props.labels} values={shape.props.values} />
        </div>
      </div>
    </BoardFrame>
  )
}

const boxProps = {
  w: T.number,
  h: T.number,
}

export class BoardQuestionShapeUtil extends BaseBoxShapeUtil<IBoardQuestionShape> {
  static override type = BOARD_QUESTION_TYPE
  static override props: RecordProps<IBoardQuestionShape> = {
    ...boxProps,
    title: T.string,
    prompt: T.string,
    source: T.string,
    status: T.literalEnum("idle", "solving", "solved", "error"),
    requestId: T.string,
    childIds: T.any as unknown as RecordProps<IBoardQuestionShape>["childIds"],
    arrowIds: T.any as unknown as RecordProps<IBoardQuestionShape>["arrowIds"],
    origin: T.literalEnum("freeform", "tracker"),
    solutionHidden: T.boolean,
    isPromptHidden: T.boolean,
  }

  override getDefaultProps(): IBoardQuestionShape["props"] {
    return {
      w: 680,
      h: QUESTION_H,
      title: "",
      prompt: "",
      source: "",
      status: "idle",
      requestId: "",
      childIds: [],
      arrowIds: [],
      origin: "freeform",
      solutionHidden: false,
      isPromptHidden: false,
    }
  }

  override canEdit() {
    return false
  }

  canDelete() {
    return false
  }

  override canResize() {
    return true
  }

  override isAspectRatioLocked() {
    return false
  }

  override onResize(shape: IBoardQuestionShape, info: Parameters<BaseBoxShapeUtil<IBoardQuestionShape>["onResize"]>[1]) {
    return resizeBox(shape, info, { minWidth: 240, minHeight: 80 })
  }

  override component(shape: IBoardQuestionShape) {
    return <QuestionBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardQuestionShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}

export class BoardStepShapeUtil extends BaseBoxShapeUtil<IBoardStepShape> {
  static override type = BOARD_STEP_TYPE
  static override props: RecordProps<IBoardStepShape> = {
    ...boxProps,
    latex: T.string,
    index: T.number,
    parentId: T.string,
    branchFromId: T.string,
    isExplanationBranch: T.boolean,
    explanationHidden: T.boolean,
    expandedGeometry: T.any as unknown as RecordProps<IBoardStepShape>["expandedGeometry"],
  }

  override getDefaultProps(): IBoardStepShape["props"] {
    return {
      w: 680,
      h: 64,
      latex: "",
      index: 0,
      parentId: "",
      branchFromId: "",
      isExplanationBranch: false,
      explanationHidden: false,
    }
  }

  override canEdit() {
    return false
  }

  canDelete() {
    return false
  }

  override canResize() {
    return true
  }

  override isAspectRatioLocked() {
    return false
  }

  override onResize(shape: IBoardStepShape, info: Parameters<BaseBoxShapeUtil<IBoardStepShape>["onResize"]>[1]) {
    if (isCollapsedBoardShape(shape.props)) return shape
    return resizeBox(shape, info, { minWidth: 180, minHeight: 40 })
  }

  override component(shape: IBoardStepShape) {
    return <StepBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardStepShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}

export class BoardResultShapeUtil extends BaseBoxShapeUtil<IBoardResultShape> {
  static override type = BOARD_RESULT_TYPE
  static override props: RecordProps<IBoardResultShape> = {
    ...boxProps,
    value: T.string,
    summary: T.string,
    parentId: T.string,
    expandedGeometry: T.any as unknown as RecordProps<IBoardResultShape>["expandedGeometry"],
  }

  override getDefaultProps(): IBoardResultShape["props"] {
    return { w: 680, h: 96, value: "", summary: "", parentId: "" }
  }

  override canEdit() {
    return false
  }

  canDelete() {
    return false
  }

  override canResize() {
    return true
  }

  override isAspectRatioLocked() {
    return false
  }

  override onResize(shape: IBoardResultShape, info: Parameters<BaseBoxShapeUtil<IBoardResultShape>["onResize"]>[1]) {
    if (isCollapsedBoardShape(shape.props)) return shape
    return resizeBox(shape, info, { minWidth: 180, minHeight: 48 })
  }

  override component(shape: IBoardResultShape) {
    return <ResultBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardResultShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}

export class BoardChartShapeUtil extends BaseBoxShapeUtil<IBoardChartShape> {
  static override type = BOARD_CHART_TYPE
  static override props: RecordProps<IBoardChartShape> = {
    ...boxProps,
    chart_type: T.literalEnum("line", "bar", "pie"),
    title: T.string,
    labels: T.any as unknown as RecordProps<IBoardChartShape>["labels"],
    values: T.any as unknown as RecordProps<IBoardChartShape>["values"],
    parentId: T.string,
    expandedGeometry: T.any as unknown as RecordProps<IBoardChartShape>["expandedGeometry"],
  }

  override getDefaultProps(): IBoardChartShape["props"] {
    return {
      w: 640,
      h: 268,
      chart_type: "line",
      title: "",
      labels: [],
      values: [],
      parentId: "",
    }
  }

  override canEdit() {
    return false
  }

  canDelete() {
    return false
  }

  override canResize() {
    return true
  }

  override isAspectRatioLocked() {
    return false
  }

  override onResize(shape: IBoardChartShape, info: Parameters<BaseBoxShapeUtil<IBoardChartShape>["onResize"]>[1]) {
    if (isCollapsedBoardShape(shape.props)) return shape
    return resizeBox(shape, info, { minWidth: 280, minHeight: 160 })
  }

  override component(shape: IBoardChartShape) {
    return <ChartBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardChartShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}

function BranchBody({ shape }: { shape: IBoardBranchShape }) {
  return (
    <BoardFrame width={shape.props.w} height={shape.props.h} overflow="visible">
      <div className="board-branch-shell h-full w-full" />
    </BoardFrame>
  )
}

export class BoardBranchShapeUtil extends BaseBoxShapeUtil<IBoardBranchShape> {
  static override type = BOARD_BRANCH_TYPE
  static override props: RecordProps<IBoardBranchShape> = {
    ...boxProps,
    parentId: T.string,
    branchFromId: T.string,
    lastRequestId: T.string,
    expandedGeometry: T.any as unknown as RecordProps<IBoardBranchShape>["expandedGeometry"],
  }

  override getDefaultProps(): IBoardBranchShape["props"] {
    return { w: 564, h: 140, parentId: "", branchFromId: "", lastRequestId: "" }
  }

  override canEdit() {
    return false
  }

  canDelete() {
    return false
  }

  override canResize() {
    return false
  }

  override isAspectRatioLocked() {
    return false
  }

  override canReceiveNewChildrenOfType(_shape: IBoardBranchShape, type: string) {
    return type === BOARD_STEP_TYPE
  }

  override hideSelectionBoundsFg() {
    return false
  }

  override component(shape: IBoardBranchShape) {
    return <BranchBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardBranchShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}

