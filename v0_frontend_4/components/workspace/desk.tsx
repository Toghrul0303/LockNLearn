"use client"

import { useRef, useState } from "react"
import dynamic from "next/dynamic"
import type { TLStore } from "tldraw"
import { FileText, Layers, Pencil, Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useDesk } from "./desk-context"
import { BottomDock } from "./bottom-dock"
import { useLanguage } from "./language-context"
import { useWhiteboard } from "./canvas/whiteboard-context"

const DeskEditor = dynamic(
  () => import("./canvas/desk-editor").then((mod) => mod.DeskEditor),
  { ssr: false },
)

export function Desk() {
  const { setEditor } = useDesk()
  const { boards, activeBoardId, activeStore, selectBoard, addBoard, closeBoard, renameBoard } =
    useWhiteboard()
  const canvasRef = useRef<HTMLDivElement>(null)
  const mountedBoard = useRef<{ store: TLStore; boardId: string } | null>(null)
  const { t } = useLanguage()
  if (activeStore && activeBoardId) {
    mountedBoard.current = { store: activeStore, boardId: activeBoardId }
  }
  const shown = activeStore && activeBoardId ? { store: activeStore, boardId: activeBoardId } : mountedBoard.current
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState("")
  const skipCommit = useRef(false)

  function beginRename(id: string, name: string) {
    skipCommit.current = false
    setEditingId(id)
    setDraftName(name)
  }

  function commitRename() {
    if (skipCommit.current) {
      skipCommit.current = false
      return
    }
    const id = editingId
    const next = draftName
    setEditingId(null)
    if (id) renameBoard(id, next)
  }

  function cancelRename() {
    skipCommit.current = true
    setEditingId(null)
  }

  return (
    <section
      aria-label={t("desk.aria")}
      className="relative flex min-w-0 flex-1 flex-col overflow-hidden"
    >
      <div className="flex items-center gap-1 border-b border-border px-3 py-2">
        <span className="mr-1 grid size-7 place-items-center rounded-lg bg-accent text-accent-foreground">
          <Layers className="size-4" aria-hidden="true" />
        </span>
        {boards.map((board) =>
          editingId === board.id ? (
            <input
              key={board.id}
              autoFocus
              value={draftName}
              aria-label={t("desk.renameBoard")}
              maxLength={80}
              onChange={(event) => setDraftName(event.target.value)}
              onBlur={commitRename}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault()
                  event.currentTarget.blur()
                } else if (event.key === "Escape") {
                  event.preventDefault()
                  cancelRename()
                }
              }}
              className="h-8 w-36 rounded-lg border border-border bg-card px-2 text-sm text-foreground outline-none"
            />
          ) : (
          <button
            key={board.id}
            type="button"
            onClick={() => selectBoard(board.id)}
            onDoubleClick={(event) => {
              event.preventDefault()
              beginRename(board.id, board.name)
            }}
            className={cn(
              "group flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors",
              activeBoardId === board.id
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
            )}
          >
            <FileText className="size-3.5" aria-hidden="true" />
            <span className="max-w-[10rem] truncate">{board.name}</span>
            {activeBoardId === board.id && (
              <span
                role="button"
                tabIndex={0}
                aria-label={t("desk.renameBoard")}
                className="grid size-4 place-items-center rounded text-muted-foreground/60 transition-colors hover:text-foreground"
                onClick={(event) => {
                  event.stopPropagation()
                  beginRename(board.id, board.name)
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault()
                    event.stopPropagation()
                    beginRename(board.id, board.name)
                  }
                }}
              >
                <Pencil className="size-3" aria-hidden="true" />
              </span>
            )}
            {activeBoardId === board.id && boards.length > 1 && (
              <span
                role="button"
                tabIndex={0}
                aria-label={t("desk.closeBoard")}
                className="grid size-4 place-items-center rounded text-muted-foreground/60 transition-colors hover:text-foreground"
                onClick={(event) => {
                  event.stopPropagation()
                  closeBoard(board.id)
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault()
                    event.stopPropagation()
                    closeBoard(board.id)
                  }
                }}
              >
                <X className="size-3.5" aria-hidden="true" />
              </span>
            )}
          </button>
          ),
        )}
        <button
          type="button"
          aria-label={t("desk.newDocument")}
          onClick={addBoard}
          className="grid size-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <Plus className="size-4" aria-hidden="true" />
        </button>
      </div>

      <div ref={canvasRef} className="relative min-h-0 flex-1 overflow-hidden bg-background">
        {shown ? (
          <DeskEditor store={shown.store} boardId={shown.boardId} onEditor={setEditor} />
        ) : null}

        <BottomDock constraintsRef={canvasRef} />
      </div>
    </section>
  )
}
