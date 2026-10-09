"use client"

import { useEffect } from "react"
import { X } from "lucide-react"
import { getIndices } from "@tldraw/utils"
import { Tldraw, type Editor, type TLShapeId, type TLShapePartial } from "tldraw"
import "tldraw/tldraw.css"
import { deskShapeUtils } from "@/components/workspace/canvas/desk-shape-utils"
import type { QuestionBookmark, QuestionClusterShape } from "@/components/workspace/memory-box-context"
import { useLanguage } from "@/components/workspace/language-context"

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

export function QuestionPreviewModal({
  bookmark,
  onClose,
}: {
  bookmark: QuestionBookmark
  onClose: () => void
}) {
  const { t } = useLanguage()

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-md">
      <button
        type="button"
        onClick={onClose}
        aria-label={t("collections.closePreview")}
        className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-xl bg-background/90 px-3 py-2 text-sm font-semibold text-foreground shadow-md"
      >
        <X className="size-4" aria-hidden="true" />
        {t("collections.closePreview")}
      </button>
      <div className="question-preview h-[min(85dvh,880px)] w-full max-w-3xl overflow-hidden rounded-2xl border border-border bg-background shadow-2xl [&_[data-board-action]]:hidden">
        <div className="relative h-full w-full">
          <Tldraw
            hideUi
            shapeUtils={deskShapeUtils}
            onMount={(editor) => {
              loadQuestionCluster(editor, bookmark.cluster)
            }}
          />
        </div>
      </div>
    </div>
  )
}
