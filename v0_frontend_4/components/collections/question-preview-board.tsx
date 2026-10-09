"use client"

import { getIndices } from "@tldraw/utils"
import { Tldraw, type Editor, type TLShapeId, type TLShapePartial } from "tldraw"
import "tldraw/tldraw.css"
import { deskShapeUtils } from "@/components/workspace/canvas/desk-shape-utils"
import type { QuestionClusterShape } from "@/components/workspace/memory-box-context"

function loadQuestionCluster(editor: Editor, cluster: QuestionClusterShape[]) {
  const present = new Set(cluster.map((shape) => shape.id))
  const pageId = editor.getCurrentPageId()
  const remaining = cluster.slice()
  const placed = new Set<string>()

  const parentIdFor = (shape: QuestionClusterShape) =>
    present.has(shape.parentId) ? shape.parentId : pageId

  while (remaining.length > 0) {
    const ready = remaining.filter((shape) => {
      const parentId = parentIdFor(shape)
      return parentId === pageId || placed.has(parentId)
    })
    const forcePage = ready.length === 0
    const batch = forcePage ? remaining : ready
    const groups = new Map<string, QuestionClusterShape[]>()
    for (const shape of batch) {
      const parentId = forcePage ? pageId : parentIdFor(shape)
      const list = groups.get(parentId) ?? []
      list.push(shape)
      groups.set(parentId, list)
    }
    const partials: TLShapePartial[] = []
    for (const [parentId, shapes] of groups) {
      const indices = getIndices(shapes.length)
      shapes.forEach((shape, index) => {
        partials.push({
          id: shape.id as TLShapeId,
          type: shape.type,
          x: shape.x,
          y: shape.y,
          parentId,
          index: indices[index],
          props: shape.props,
        } as TLShapePartial)
      })
    }
    editor.createShapes(partials)
    for (const shape of batch) placed.add(shape.id)
    const batchIds = new Set(batch.map((shape) => shape.id))
    for (let i = remaining.length - 1; i >= 0; i -= 1) {
      if (batchIds.has(remaining[i].id)) remaining.splice(i, 1)
    }
    if (forcePage) break
  }

  const bounds = editor.getShapesPageBounds(cluster.map((shape) => shape.id as TLShapeId))
  if (bounds) editor.zoomToBounds(bounds, { inset: 64 })
  editor.updateInstanceState({ isReadonly: true })
}

export function QuestionPreviewBoard({ cluster }: { cluster: QuestionClusterShape[] }) {
  return (
    <Tldraw
      hideUi
      shapeUtils={deskShapeUtils}
      onMount={(editor) => {
        loadQuestionCluster(editor, cluster)
      }}
    />
  )
}
