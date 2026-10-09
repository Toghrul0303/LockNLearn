import type { TLShape, TLShapeId } from "tldraw"

export type ExpandedGeometry = { x: number; y: number; w: number; h: number }

declare module "@tldraw/tlschema" {
  interface TLGlobalShapePropsMap {
    "board-question": {
      w: number
      h: number
      /** Short header — task_summary / tracker badge label. */
      title: string
      /** Verbatim body text (raw_pdf_text) shown under the header. */
      prompt: string
      source: string
      status: "idle" | "solving" | "solved" | "error"
      requestId: string
      childIds: string[]
      arrowIds: string[]
      origin: "freeform" | "tracker"
      solutionHidden: boolean
      isPromptHidden: boolean
    }
    "board-step": {
      w: number
      h: number
      latex: string
      index: number
      parentId: string
      /** "" for the main solution column; the anchor step id for a lateral Why/How branch. */
      branchFromId: string
      isExplanationBranch: boolean
      explanationHidden: boolean
      expandedGeometry?: ExpandedGeometry
    }
    "board-result": {
      w: number
      h: number
      value: string
      summary: string
      parentId: string
      expandedGeometry?: ExpandedGeometry
    }
    "board-chart": {
      w: number
      h: number
      chart_type: "line" | "bar" | "pie"
      title: string
      labels: string[]
      values: number[]
      parentId: string
      expandedGeometry?: ExpandedGeometry
    }
    "board-figure": {
      w: number
      h: number
      imageUrl: string
      index: number
      parentId: string
      expandedGeometry?: ExpandedGeometry
    }
    "board-branch": {
      w: number
      h: number
      /** Question shape id (logical cluster parent). */
      parentId: string
      /** Anchor board-step this explanation module belongs to. */
      branchFromId: string
      /** requestId of the explanation turn currently occupying this
       * container — a new Why/How request on the same anchor replaces
       * (rather than appends alongside) whatever this container last held. */
      lastRequestId: string
      expandedGeometry?: ExpandedGeometry
    }
  }
}

export const BOARD_QUESTION_TYPE = "board-question" as const
export const BOARD_STEP_TYPE = "board-step" as const
export const BOARD_RESULT_TYPE = "board-result" as const
export const BOARD_CHART_TYPE = "board-chart" as const
export const BOARD_FIGURE_TYPE = "board-figure" as const
export const BOARD_BRANCH_TYPE = "board-branch" as const

/** Below this size a shape is treated as store-collapsed (parked, not just hidden). */
export const COLLAPSED_CHILD = 1

export type BoardQuestionStatus = "idle" | "solving" | "solved" | "error"
export type BoardQuestionOrigin = "freeform" | "tracker"

export type BoardQuestionShape = TLShape<"board-question">
export type BoardStepShape = TLShape<"board-step">
export type BoardResultShape = TLShape<"board-result">
export type BoardChartShape = TLShape<"board-chart">
export type BoardFigureShape = TLShape<"board-figure">
export type BoardBranchShape = TLShape<"board-branch">

export function isBoardQuestionShape(shape: { type: string }): shape is BoardQuestionShape {
  return shape.type === BOARD_QUESTION_TYPE
}

export function isBoardStepShape(shape: { type: string }): shape is BoardStepShape {
  return shape.type === BOARD_STEP_TYPE
}

export function isBoardResultShape(shape: { type: string }): shape is BoardResultShape {
  return shape.type === BOARD_RESULT_TYPE
}

export function isBoardChartShape(shape: { type: string }): shape is BoardChartShape {
  return shape.type === BOARD_CHART_TYPE
}

export function isBoardFigureShape(shape: { type: string }): shape is BoardFigureShape {
  return shape.type === BOARD_FIGURE_TYPE
}

export function isBoardBranchShape(shape: { type: string }): shape is BoardBranchShape {
  return shape.type === BOARD_BRANCH_TYPE
}

export function asShapeIdList(value: unknown): TLShapeId[] {
  if (!Array.isArray(value)) return []
  return value.filter((id): id is TLShapeId => typeof id === "string") as TLShapeId[]
}

/** True once a shape has been store-collapsed (parked at 1x1). */
export function isCollapsedBoardShape(props: { w: number; h: number }): boolean {
  return Number(props.w) <= COLLAPSED_CHILD && Number(props.h) <= COLLAPSED_CHILD
}

/** Reads `explanationHidden`, tolerant of the earlier `branchHidden` field name. */
export function stepExplanationHidden(shape: BoardStepShape): boolean {
  const props = shape.props as BoardStepShape["props"] & { branchHidden?: boolean }
  return Boolean(props.explanationHidden ?? props.branchHidden)
}
