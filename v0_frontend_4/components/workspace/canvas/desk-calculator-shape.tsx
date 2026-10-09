"use client"

import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
  createShapeId,
  useEditor,
  type Editor,
  type RecordProps,
  type TLShape,
  type TLShapeId,
} from "tldraw"
import { GripHorizontal } from "lucide-react"
import { CalculatorKeypad } from "../calculator-keypad"

declare module "@tldraw/tlschema" {
  interface TLGlobalShapePropsMap {
    "desk-calculator": {
      w: number
      h: number
    }
  }
}

export const DESK_CALCULATOR_TYPE = "desk-calculator" as const
export type IDeskCalculatorShape = TLShape<"desk-calculator">

/** Natural size of `CalculatorKeypad` (fixed 220px card) plus the grip strip. */
const CALC_W = 220
const CALC_H = 360
const GRIP_H = 24

export function isDeskCalculatorShape(shape: { type: string }): shape is IDeskCalculatorShape {
  return shape.type === DESK_CALCULATOR_TYPE
}

export function findCalculatorShape(editor: Editor): IDeskCalculatorShape | null {
  return (editor.getCurrentPageShapes().find(isDeskCalculatorShape) as IDeskCalculatorShape) ?? null
}

/** Spawns at the current viewport center — wherever the user is looking,
 * however far they've panned or zoomed — rather than a fixed world point. */
export function spawnCalculatorShape(editor: Editor): TLShapeId {
  const viewport = editor.getViewportPageBounds()
  const id = createShapeId()
  editor.createShape({
    id,
    type: DESK_CALCULATOR_TYPE,
    x: viewport.x + viewport.w / 2 - CALC_W / 2,
    y: viewport.y + viewport.h / 2 - CALC_H / 2,
    props: { w: CALC_W, h: CALC_H },
  })
  return id
}

/** Toggle helper shared by the toolbar button: removes the calculator if one
 * is already on the canvas, otherwise spawns + selects a fresh one. */
export function toggleCalculatorShape(editor: Editor): void {
  const existing = findCalculatorShape(editor)
  if (existing) {
    editor.deleteShapes([existing.id])
    return
  }
  const id = spawnCalculatorShape(editor)
  editor.select(id)
}

function DeskCalculatorBody({ shape }: { shape: IDeskCalculatorShape }) {
  const editor = useEditor()

  return (
    // `.board-node` (used by the math cards) defaults to `pointer-events:
    // none` so only opted-in text zones are clickable/selectable — reusing
    // it here silently ate every keypad button's clicks. `.desk-calculator-shape`
    // is `pointer-events: all` instead (same pattern as `.desk-card-shape`),
    // so the keypad is interactive by default and only the grip strip below
    // is left to pass drags through to tldraw.
    <HTMLContainer
      className="desk-calculator-shape"
      style={{ width: shape.props.w, height: shape.props.h, overflow: "visible" }}
    >
      <div className="flex h-full w-full flex-col items-center">
        {/* Dedicated drag handle: CalculatorKeypad's own root stops pointerdown
         * propagation (so button clicks don't drag the shape), so grabbing the
         * card itself would never reach tldraw. This strip sits outside that
         * boundary and is left un-stopped, giving the shape a natural handle. */}
        <div
          aria-hidden="true"
          title="Drag to move"
          className="relative z-10 -mb-1 flex h-6 w-16 shrink-0 cursor-grab items-center justify-center rounded-t-lg border border-b-0 border-border bg-popover/95 text-muted-foreground shadow-sm active:cursor-grabbing"
          style={{ height: GRIP_H }}
        >
          <GripHorizontal className="size-3.5" aria-hidden="true" />
        </div>
        <CalculatorKeypad onClose={() => editor.deleteShapes([shape.id])} />
      </div>
    </HTMLContainer>
  )
}

export class DeskCalculatorShapeUtil extends BaseBoxShapeUtil<IDeskCalculatorShape> {
  static override type = DESK_CALCULATOR_TYPE
  static override props: RecordProps<IDeskCalculatorShape> = {
    w: T.number,
    h: T.number,
  }

  override getDefaultProps(): IDeskCalculatorShape["props"] {
    return { w: CALC_W, h: CALC_H }
  }

  override canEdit() {
    return false
  }

  override canResize() {
    return false
  }

  override component(shape: IDeskCalculatorShape) {
    return <DeskCalculatorBody shape={shape} />
  }

  override getIndicatorPath(shape: IDeskCalculatorShape) {
    const path = new Path2D()
    path.rect(0, 0, shape.props.w, shape.props.h)
    return path
  }
}
