"use client"

import { useCallback, useEffect, useRef } from "react"
import dynamic from "next/dynamic"
import {
  DefaultContextMenu,
  DefaultToolbar,
  DrawToolbarItem,
  HandToolbarItem,
  SelectToolbarItem,
  TextToolbarItem,
  TldrawUiToolbarButton,
  useEditor,
  useValue,
  type Editor,
  type TLComponents,
  type TLShape,
  type TLStore,
  type TLUiContextMenuProps,
  type TLUiOverrides,
} from "tldraw"
import { Calculator, CircleHelp } from "lucide-react"
import { DESK_CAMERA_EVENT } from "@/lib/desk-events"
import { useMode } from "../mode-context"
import { useLanguage } from "../language-context"
import { useExplainMode } from "../explain-mode-context"

const DeskTldraw = dynamic(() => import("./desk-tldraw").then((mod) => mod.DeskTldraw), {
  ssr: false,
})
import { findCalculatorShape, toggleCalculatorShape } from "./desk-calculator-shape"
import { DESK_CARD_TYPE } from "./desk-card-shape"
import {
  BOARD_BRANCH_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_QUESTION_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_STEP_TYPE,
} from "./board-constants"

const GENERATED_SHAPE_TYPES = new Set<string>([
  BOARD_QUESTION_TYPE,
  BOARD_STEP_TYPE,
  BOARD_RESULT_TYPE,
  BOARD_CHART_TYPE,
  BOARD_FIGURE_TYPE,
  BOARD_BRANCH_TYPE,
  DESK_CARD_TYPE,
])

function selectionIsGenerated(editor: Editor) {
  const walk = (shape: TLShape): boolean => {
    if (GENERATED_SHAPE_TYPES.has(shape.type)) return true
    if (shape.type === "arrow") {
      return editor.getBindingsInvolvingShape(shape.id).some((binding) => {
        const to = editor.getShape(binding.toId)
        return Boolean(to && to.id !== shape.id && GENERATED_SHAPE_TYPES.has(to.type))
      })
    }
    if (shape.type !== "group") return false
    for (const childId of editor.getSortedChildIdsForParent(shape.id)) {
      const child = editor.getShape(childId)
      if (child && walk(child)) return true
    }
    return false
  }
  return editor.getSelectedShapes().some((shape) => walk(shape))
}

function DeskGeneratedContextMenu(props: TLUiContextMenuProps) {
  const editor = useEditor()
  const disabled = useValue(
    "generated-context-menu",
    () => selectionIsGenerated(editor),
    [editor],
  )
  return <DefaultContextMenu {...props} disabled={disabled} />
}

function SlimToolbar() {
  const editor = useEditor()
  const { t } = useLanguage()
  const { armed, setArmed } = useExplainMode()
  const calculatorOpen = useValue(
    "calculator-shape-present",
    () => Boolean(findCalculatorShape(editor)),
    [editor],
  )

  return (
    <DefaultToolbar orientation="vertical" minItems={6} maxItems={6}>
      <SelectToolbarItem />
      <HandToolbarItem />
      <DrawToolbarItem />
      <TextToolbarItem />
      <TldrawUiToolbarButton
        type="tool"
        title={t("chat.explain")}
        tooltip={t("chat.explain")}
        isActive={armed}
        onClick={() => setArmed(!armed)}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <CircleHelp className="size-[18px] shrink-0 stroke-current" aria-hidden="true" />
      </TldrawUiToolbarButton>
      <TldrawUiToolbarButton
        type="tool"
        title={t("desk.calculator")}
        tooltip={t("desk.calculator")}
        isActive={calculatorOpen}
        onClick={() => toggleCalculatorShape(editor)}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <Calculator className="size-[18px] shrink-0 stroke-current" aria-hidden="true" />
      </TldrawUiToolbarButton>
    </DefaultToolbar>
  )
}

const deskTldrawComponents: TLComponents = {
  Toolbar: SlimToolbar,
  ContextMenu: DeskGeneratedContextMenu,
  MainMenu: null,
  HelpMenu: null,
  PageMenu: null,
  Minimap: null,
  StylePanel: null,
  NavigationPanel: null,
  DebugPanel: null,
  DebugMenu: null,
  MenuPanel: null,
  TopPanel: null,
  SharePanel: null,
  KeyboardShortcutsDialog: null,
  HelperButtons: null,
  QuickActions: null,
  ImageToolbar: null,
  VideoToolbar: null,
  PeopleMenu: null,
  CursorChatBubble: null,
  ZoomMenu: null,
}

const deskUiOverrides: TLUiOverrides = {
  tools(_editor, tools) {
    const keep = new Set(["select", "hand", "draw", "text"])
    for (const id of Object.keys(tools)) {
      if (!keep.has(id)) delete tools[id]
    }
    return tools
  },
  actions(editor, actions) {
    const selectionIsGeneratedNow = () => selectionIsGenerated(editor)
    for (const id of ["delete", "ungroup"] as const) {
      const action = actions[id]
      if (!action) continue
      const prev = action.onSelect.bind(action)
      action.onSelect = (source) => {
        if (selectionIsGeneratedNow()) return
        return prev(source)
      }
    }
    return actions
  },
}

export function DeskEditor({
  store,
  boardId,
  onEditor,
}: {
  store: TLStore
  boardId: string
  onEditor: (editor: Editor | null) => void
}) {
  const { dark } = useMode()
  const onEditorRef = useRef(onEditor)
  onEditorRef.current = onEditor

  const handleMount = useCallback((editor: Editor) => {
    onEditorRef.current(editor)
    const unsub = editor.store.listen(
      () => {
        window.dispatchEvent(new Event(DESK_CAMERA_EVENT))
      },
      { source: "user", scope: "session" },
    )
    return () => {
      unsub()
      onEditorRef.current(null)
    }
  }, [])

  useEffect(() => {
    return () => onEditorRef.current(null)
  }, [boardId])

  return (
    <div className="desk-tldraw absolute inset-0">
      <DeskTldraw
        key={boardId}
        store={store}
        components={deskTldrawComponents}
        overrides={deskUiOverrides}
        onMount={handleMount}
        colorScheme={dark ? "dark" : "light"}
      />
    </div>
  )
}
