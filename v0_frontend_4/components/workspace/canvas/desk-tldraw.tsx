"use client"

import { Tldraw, type Editor, type TLComponents, type TLStore, type TLUiOverrides } from "tldraw"
import "tldraw/tldraw.css"
import { deskShapeUtils } from "./desk-shape-utils"

export function DeskTldraw({
  store,
  components,
  overrides,
  onMount,
  colorScheme,
}: {
  store: TLStore
  components: TLComponents
  overrides: TLUiOverrides
  onMount: (editor: Editor) => void | (() => void)
  colorScheme: "dark" | "light"
}) {
  return (
    <Tldraw
      store={store}
      shapeUtils={deskShapeUtils}
      components={components}
      overrides={overrides}
      onMount={onMount}
      colorScheme={colorScheme}
    />
  )
}
