"use client"

import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
  createShapeId,
  resizeBox,
  useEditor,
  useValue,
  type Editor,
  type RecordProps,
  type TLShape,
  type TLShapeId,
} from "tldraw"
import { Bookmark, BookmarkCheck, Calculator, PieChart, TrendingUp, Box, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { DynamicDeskWidget } from "../dynamic-desk-widget"
import { useDesk, type DeskItem } from "../desk-context"
import { useMemoryBox } from "../memory-box-context"
import { useLanguage } from "../language-context"

declare module "@tldraw/tlschema" {
  interface TLGlobalShapePropsMap {
    "desk-card": {
      w: number
      h: number
      item: DeskItem
    }
  }
}

export const DESK_CARD_TYPE = "desk-card" as const
export const MAX_DESK_CARDS = 8
const COMPACT_ZOOM = 0.5

export type IDeskCardShape = TLShape<"desk-card">

export function deskCardShapeId(itemId: string): TLShapeId {
  return createShapeId(itemId)
}

export function isDeskCardShape(shape: { type: string }): shape is IDeskCardShape {
  return shape.type === DESK_CARD_TYPE
}

export function cardSizeFor(item: DeskItem): { w: number; h: number } {
  const visual = item.type === "chart" || item.type === "diagram"
  return visual ? { w: 420, h: 320 } : { w: 380, h: 280 }
}

/** Strip `undefined` (and other non-JSON values) so tldraw's store can persist the payload. */
export function serializeDeskItem(item: DeskItem): DeskItem {
  return JSON.parse(JSON.stringify(item)) as DeskItem
}

export function readDeskItem(raw: unknown): DeskItem | null {
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw) as DeskItem
    } catch {
      return null
    }
  }
  if (raw && typeof raw === "object" && "type" in raw && "id" in raw) {
    return raw as DeskItem
  }
  return null
}

export function listDeskItemsFromEditor(editor: Editor): DeskItem[] {
  return editor
    .getCurrentPageShapes()
    .filter(isDeskCardShape)
    .sort((a, b) => (a.index < b.index ? -1 : 1))
    .map((shape) => readDeskItem(shape.props.item))
    .filter((item): item is DeskItem => item != null)
}

export function createDeskCardShape(editor: Editor, item: DeskItem) {
  const existing = editor.getCurrentPageShapes().filter(isDeskCardShape)
  const { w, h } = cardSizeFor(item)
  const n = existing.length
  editor.createShape({
    id: deskCardShapeId(item.id),
    type: DESK_CARD_TYPE,
    x: 72 + (n % 4) * 36,
    y: 96 + n * 28,
    props: { w, h, item: serializeDeskItem(item) },
  })
  const after = editor.getCurrentPageShapes().filter(isDeskCardShape)
  if (after.length > MAX_DESK_CARDS) {
    const drop = [...after]
      .sort((a, b) => (a.index < b.index ? -1 : 1))
      .slice(0, after.length - MAX_DESK_CARDS)
    editor.deleteShapes(drop.map((shape) => shape.id))
  }
}

function cardTitle(item: DeskItem): string {
  if (item.type === "calculation") return item.headerTitle?.trim() || "Math & Calculation"
  return item.title?.trim() || (item.type === "diagram" ? "Diagram" : "Chart")
}

function CardIcon({ item }: { item: DeskItem }) {
  if (item.type === "calculation") return <Calculator className="size-4" aria-hidden="true" />
  if (item.type === "diagram") return <Box className="size-4" aria-hidden="true" />
  if (item.type === "chart" && item.chart_type === "pie") {
    return <PieChart className="size-4" aria-hidden="true" />
  }
  return <TrendingUp className="size-4" aria-hidden="true" />
}

function DeskCardBody({ shape }: { shape: IDeskCardShape }) {
  const editor = useEditor()
  const compact = useValue(
    "desk-card-compact",
    () => {
      const zoom = editor.getZoomLevel()
      if (zoom < COMPACT_ZOOM) return true
      const pageBounds = editor.getShapePageBounds(shape.id)
      if (!pageBounds) return true
      const viewport = editor.getViewportPageBounds()
      return !viewport.collides(pageBounds)
    },
    [editor, shape],
  )

  const { removeDeskItem, patchDeskItem } = useDesk()
  const { saveBookmark } = useMemoryBox()
  const { t } = useLanguage()
  const item = readDeskItem(shape.props.item)
  if (!item) return null

  const handleSave = (event: React.MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    saveBookmark(item)
    patchDeskItem(item.id, { bookmarked: true })
  }

  const handleRemove = (event: React.MouseEvent) => {
    event.stopPropagation()
    event.preventDefault()
    removeDeskItem(item.id)
  }

  return (
    <HTMLContainer
      className="desk-card-shape relative overflow-hidden rounded-2xl"
      style={{ width: shape.props.w, height: shape.props.h }}
    >
      <span
        className="desk-card-drag-strip absolute inset-x-0 top-0 z-20 h-7 cursor-grab"
        data-desk-card-drag=""
      />
      <div className="absolute right-2 top-1.5 z-30 flex items-center gap-1">
        <button
          type="button"
          aria-label={item.bookmarked ? t("desk.savedBookmark") : t("desk.saveBookmark")}
          disabled={item.bookmarked}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={handleSave}
          className={cn(
            "grid size-6 place-items-center rounded-md bg-popover/90 shadow-sm backdrop-blur-sm transition-colors",
            item.bookmarked ? "text-emerald-500" : "text-muted-foreground hover:text-primary",
          )}
        >
          {item.bookmarked ? (
            <BookmarkCheck className="size-3.5" aria-hidden="true" />
          ) : (
            <Bookmark className="size-3.5" aria-hidden="true" />
          )}
        </button>
        <button
          type="button"
          aria-label={t("desk.removeCard")}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={handleRemove}
          className="grid size-6 place-items-center rounded-md bg-popover/90 text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:text-destructive"
        >
          <X className="size-3.5" aria-hidden="true" />
        </button>
      </div>
      {compact ? (
        <div
          className="desk-artifact flex h-full items-center gap-2 rounded-2xl border border-border bg-card/90 px-4 shadow-sm"
          data-highlight-source=""
        >
          <span className="bg-brand-gradient grid size-8 shrink-0 place-items-center rounded-lg text-white">
            <CardIcon item={item} />
          </span>
          <p className="min-w-0 truncate font-display text-sm font-semibold">{cardTitle(item)}</p>
        </div>
      ) : (
        <div
          className="desk-artifact h-full min-h-0"
          data-highlight-source=""
          style={{ pointerEvents: "all", userSelect: "text" }}
          onPointerDown={(event) => {
            const target = event.target as HTMLElement | null
            if (target?.closest("[data-desk-card-drag]")) return
            event.stopPropagation()
          }}
        >
          <DynamicDeskWidget update={item} />
        </div>
      )}
    </HTMLContainer>
  )
}

export class DeskCardShapeUtil extends BaseBoxShapeUtil<IDeskCardShape> {
  static override type = DESK_CARD_TYPE
  static override props: RecordProps<IDeskCardShape> = {
    w: T.number,
    h: T.number,
    // Nested DeskItem payloads include optional/undefined fields that
    // `T.jsonValue` rejects. `T.any` lets the card persist without crashing.
    item: T.any as unknown as RecordProps<IDeskCardShape>["item"],
  }

  override getDefaultProps(): IDeskCardShape["props"] {
    return {
      w: 380,
      h: 280,
      item: {
        id: "placeholder",
        type: "calculation",
        value: "",
        summary: "",
      },
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

  override canScroll() {
    return true
  }

  override onResize(shape: IDeskCardShape, info: Parameters<BaseBoxShapeUtil<IDeskCardShape>["onResize"]>[1]) {
    return resizeBox(shape, info, { minWidth: 280, minHeight: 180 })
  }

  override component(shape: IDeskCardShape) {
    return <DeskCardBody shape={shape} />
  }

  override getIndicatorPath(shape: IDeskCardShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}
