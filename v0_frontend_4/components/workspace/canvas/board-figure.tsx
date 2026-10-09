"use client"

import { useEffect, useRef, useState, type SyntheticEvent } from "react"
import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
  resizeBox,
  useEditor,
  useValue,
  type RecordProps,
  type TLShapeId,
} from "tldraw"
import {
  BOARD_FIGURE_TYPE,
  isBoardFigureShape,
  isCollapsedBoardShape,
} from "./board-constants"
import { groupQuestionCluster, reflowMainColumn } from "./canvas-layout"

export type IBoardFigureShape = import("tldraw").TLShape<"board-figure">

const COMPACT_ZOOM = 0.4
const CARD_CLASS =
  "box-border flex w-full flex-col gap-2 rounded-xl border border-black/[0.08] bg-card px-3 py-4 shadow-[0_1px_0_rgba(0,0,0,0.03)]"

function useCompact(shapeId: TLShapeId) {
  const editor = useEditor()
  return useValue(
    "board-figure-compact",
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

function FigureBody({ shape }: { shape: IBoardFigureShape }) {
  const editor = useEditor()
  const compact = useCompact(shape.id)
  const imgRef = useRef<HTMLImageElement>(null)
  const [failed, setFailed] = useState(false)
  const inlined = Boolean(shape.props.parentId)

  const applyImageSize = (img: HTMLImageElement) => {
    if (!img.naturalWidth || !img.naturalHeight) return
    const current = editor.getShape(shape.id)
    if (!current || !isBoardFigureShape(current) || isCollapsedBoardShape(current.props)) return
    const chrome = 32
    const nextH = Math.max(
      80,
      Math.ceil((img.naturalHeight / img.naturalWidth) * current.props.w) + chrome,
    )
    if (Math.abs(nextH - current.props.h) > 4) {
      editor.updateShape({
        id: current.id,
        type: BOARD_FIGURE_TYPE,
        props: { h: nextH },
      })
    }
    const parentId = current.props.parentId as TLShapeId
    if (parentId) {
      reflowMainColumn(editor, parentId)
      groupQuestionCluster(editor, parentId)
    }
  }

  const onLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    applyImageSize(event.currentTarget)
  }

  useEffect(() => {
    if (inlined) return
    const img = imgRef.current
    if (img?.complete && img.naturalWidth) applyImageSize(img)
    // Height must wait for the bitmap. Cached images fire complete without onLoad.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor, shape.id, shape.props.imageUrl, shape.props.w, inlined])

  // Parented crops are painted inside the question card; this shape is a 1×1 holder.
  if (inlined) {
    return (
      <HTMLContainer
        className="board-node"
        style={{ width: shape.props.w, height: shape.props.h, pointerEvents: "none" }}
      />
    )
  }

  if (compact) {
    return (
      <HTMLContainer className="board-node" style={{ width: shape.props.w, height: shape.props.h }}>
        <p className="truncate px-1 text-sm text-muted-foreground">Figure</p>
      </HTMLContainer>
    )
  }

  return (
    <HTMLContainer
      className="board-node"
      style={{ width: shape.props.w, height: shape.props.h, overflow: "visible" }}
    >
      <div className={CARD_CLASS}>
        {failed ? (
          <p className="text-sm text-muted-foreground">Figure could not be loaded.</p>
        ) : (
          <img
            ref={imgRef}
            src={shape.props.imageUrl}
            alt=""
            draggable={false}
            onLoad={onLoad}
            onError={() => setFailed(true)}
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        )}
      </div>
    </HTMLContainer>
  )
}

export class BoardFigureShapeUtil extends BaseBoxShapeUtil<IBoardFigureShape> {
  static override type = BOARD_FIGURE_TYPE
  static override props: RecordProps<IBoardFigureShape> = {
    w: T.number,
    h: T.number,
    imageUrl: T.string,
    index: T.number,
    parentId: T.string,
    expandedGeometry: T.any as unknown as RecordProps<IBoardFigureShape>["expandedGeometry"],
  }

  override getDefaultProps(): IBoardFigureShape["props"] {
    return { w: 680, h: 160, imageUrl: "", index: 0, parentId: "" }
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

  override onResize(shape: IBoardFigureShape, info: Parameters<BaseBoxShapeUtil<IBoardFigureShape>["onResize"]>[1]) {
    if (isCollapsedBoardShape(shape.props)) return shape
    return resizeBox(shape, info, { minWidth: 180, minHeight: 64 })
  }

  override component(shape: IBoardFigureShape) {
    return <FigureBody shape={shape} />
  }

  override getIndicatorPath(shape: IBoardFigureShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}
