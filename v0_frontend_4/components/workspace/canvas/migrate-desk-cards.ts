import type { Editor } from "tldraw"
import type { DeskItem } from "../desk-context"
import { isDeskCardShape, readDeskItem } from "./desk-card-shape"
import {
  beginSolveSession,
  spawnChartNode,
  spawnQuestionShape,
  spawnResultNode,
  spawnStepNode,
} from "./canvas-layout"
import { spawnDiagramPrimitives } from "./spawn-diagram"
import { formatBoardValue } from "./board-math"

function promptFromCard(item: DeskItem): string {
  if (item.type === "calculation") {
    return (
      item.headerDescription?.trim() ||
      item.summary?.trim() ||
      formatBoardValue(item.value) ||
      "Problem"
    )
  }
  if (item.type === "chart") return item.title?.trim() || "Chart"
  return item.title?.trim() || "Diagram"
}

function migrateOne(editor: Editor, item: DeskItem) {
  if (item.type === "calculation") {
    const questionId = spawnQuestionShape(
      editor,
      promptFromCard(item),
      item.headerTitle?.trim() || "",
    )
    const session = beginSolveSession(editor, questionId, item.id)
    if (!session) return
    const steps = item.steps ?? []
    steps.forEach((latex, index) => {
      if (latex.trim()) spawnStepNode(editor, session, latex, index)
    })
    spawnResultNode(editor, session, formatBoardValue(item.value), item.summary || "")
    return
  }
  if (item.type === "chart") {
    spawnChartNode(editor, null, { x: 72, y: 96 }, {
      chart_type: item.chart_type,
      title: item.title || "",
      labels: item.labels || [],
      values: item.values || [],
    })
    return
  }
  spawnDiagramPrimitives(
    editor,
    {
      title: item.title,
      width: item.width,
      height: item.height,
      elements: item.elements,
    },
    { x: 72, y: 96 },
  )
}

/** One-shot: turn leftover `desk-card` snapshots into the board-node family. */
export function migrateDeskCards(editor: Editor) {
  const cards = editor.getCurrentPageShapes().filter(isDeskCardShape)
  if (cards.length === 0) return
  const items = cards
    .map((shape) => readDeskItem(shape.props.item))
    .filter((item): item is DeskItem => item != null)
  editor.deleteShapes(cards.map((shape) => shape.id))
  for (const item of items) migrateOne(editor, item)
}

/** Session records that still store the old card pile. */
export function spawnLegacyDeskItems(editor: Editor, items: DeskItem[]) {
  for (const item of items) migrateOne(editor, item)
}
