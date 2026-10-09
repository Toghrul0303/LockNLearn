"use client"

import { useRef, useState } from "react"
import { useRouter } from "next/navigation"
import {
  ChevronDown,
  ChevronsLeft,
  ChevronsRight,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { MEMORY_FOLDERS, type Chapter } from "./data"
import { STATUS_STYLES, nextQuestionStatus, useTaskTracker } from "./task-tracker-context"
import { useSession } from "./session-context"
import { useMemoryBox } from "./memory-box-context"
import { useLanguage } from "./language-context"

/** Folders that route to a dedicated tab on the `/collections` page. */
const FOLDER_ROUTES: Record<string, string> = {
  formula: "/collections?tab=formulas",
  bookmarks: "/collections?tab=bookmarks",
}

const COMING_SOON_FOLDERS = new Set(["graphs", "summaries"])

export function LeftSidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean
  onToggle: () => void
}) {
  const { resetSession, sessions, threadId, selectSession, deleteSession, patchSession } = useSession()
  const {
    savedFormulas,
    savedGraphs,
    savedBookmarks,
  } = useMemoryBox()
  const router = useRouter()
  const { t } = useLanguage()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftTitle, setDraftTitle] = useState("")
  const skipCommit = useRef(false)

  function beginRename(id: string, title: string) {
    skipCommit.current = false
    setEditingId(id)
    setDraftTitle(title)
  }

  function commitRename() {
    if (skipCommit.current) {
      skipCommit.current = false
      return
    }
    const id = editingId
    const next = draftTitle.trim()
    setEditingId(null)
    if (!id || !next) return
    const current = sessions.find((session) => session.id === id)
    if (!current || current.title === next) return
    patchSession(id, { title: next })
  }

  function cancelRename() {
    skipCommit.current = true
    setEditingId(null)
  }

  const folderCount = (folderId: string, fallback: number) => {
    if (folderId === "formula") return savedFormulas?.length ?? fallback
    if (folderId === "graphs") return savedGraphs?.length ?? fallback
    if (folderId === "bookmarks") return savedBookmarks?.length ?? fallback
    return fallback
  }

  return (
    <aside
      className={cn(
        "relative z-20 flex shrink-0 flex-col border-r border-border bg-sidebar text-sidebar-foreground transition-all duration-500 ease-out",
        collapsed ? "w-16" : "w-72",
      )}
    >
      {/* Header / session buttons */}
      <div className="flex flex-col gap-2 p-3">
        <div className="flex items-center justify-between">
          {!collapsed && (
            <span className="font-display text-sm font-semibold text-muted-foreground">
              {t("sidebar.workspace")}
            </span>
          )}
          <button
            type="button"
            onClick={onToggle}
            aria-label={collapsed ? t("sidebar.expand") : t("sidebar.collapse")}
            className="grid size-8 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
          >
            {collapsed ? (
              <ChevronsRight className="size-4" aria-hidden="true" />
            ) : (
              <ChevronsLeft className="size-4" aria-hidden="true" />
            )}
          </button>
        </div>

        <button
          type="button"
          onClick={resetSession}
          className={cn(
            "bg-brand-gradient ring-brand-glow flex h-10 items-center gap-2 rounded-xl px-3 text-sm font-semibold text-white transition-transform hover:-translate-y-px",
            collapsed && "justify-center px-0",
          )}
        >
          <Plus className="size-4" aria-hidden="true" />
          {!collapsed && t("sidebar.newSession")}
        </button>
      </div>

      {collapsed ? (
        <CollapsedRail />
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-4 px-3 pb-4">
          <Section title={t("sidebar.mySessions")} className="shrink-0">
            {sessions.length === 0 ? (
              <p className="px-2.5 py-2 text-xs leading-relaxed text-muted-foreground">
                {t("sidebar.emptySessions")}
              </p>
            ) : (
              <ul className="scroll-slim max-h-40 space-y-1 overflow-y-auto pr-0.5">
                {[...sessions]
                  .sort((a, b) => b.lastActiveAt - a.lastActiveAt)
                  .map((s) => {
                    const active = s.id === threadId
                    return (
                      <li key={s.id} className="group relative">
                        {editingId === s.id ? (
                          <input
                            autoFocus
                            value={draftTitle}
                            aria-label={t("sidebar.renameSession")}
                            maxLength={80}
                            onChange={(event) => setDraftTitle(event.target.value)}
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
                            className="h-9 w-full rounded-lg border border-sidebar-border bg-sidebar-accent px-2.5 text-sm text-sidebar-accent-foreground outline-none"
                          />
                        ) : (
                          <button
                            type="button"
                            onClick={() => selectSession(s.id)}
                            onDoubleClick={(event) => {
                              event.preventDefault()
                              beginRename(s.id, s.title)
                            }}
                            className={cn(
                              "flex w-full items-center gap-2 rounded-lg py-2 pl-2.5 pr-16 text-left text-sm transition-colors",
                              active
                                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                                : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                            )}
                          >
                            <span
                              className={cn(
                                "size-1.5 shrink-0 rounded-full",
                                active ? "bg-primary" : "bg-muted-foreground/40",
                              )}
                            />
                            <span className="truncate">{s.title}</span>
                          </button>
                        )}
                        {editingId !== s.id && (
                        <div className="pointer-events-none absolute inset-y-0 right-1 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                          <button
                            type="button"
                            aria-label={t("sidebar.renameSession")}
                            onClick={(event) => {
                              event.stopPropagation()
                              beginRename(s.id, s.title)
                            }}
                            className="pointer-events-auto grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
                          >
                            <Pencil className="size-3.5" aria-hidden="true" />
                          </button>
                          <button
                            type="button"
                            aria-label={t("sidebar.deleteSession", { title: s.title })}
                            onClick={(event) => {
                              event.stopPropagation()
                              deleteSession(s.id)
                            }}
                            className="pointer-events-auto grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                          >
                            <Trash2 className="size-3.5" aria-hidden="true" />
                          </button>
                        </div>
                        )}
                      </li>
                    )
                  })}
              </ul>
            )}
          </Section>

          <TaskTracker />

          <Section title={t("sidebar.memoryBox")} className="shrink-0">
            <div className="grid grid-cols-2 gap-2">
              {MEMORY_FOLDERS.map((folder) => {
                const Icon = folder.icon
                const comingSoon = COMING_SOON_FOLDERS.has(folder.id)
                const route = comingSoon ? undefined : FOLDER_ROUTES[folder.id]
                return (
                  <button
                    key={folder.id}
                    type="button"
                    onClick={route ? () => router.push(route) : undefined}
                    disabled={!route}
                    title={comingSoon ? t("collections.comingSoon") : undefined}
                    className={cn(
                      "group flex flex-col gap-2 rounded-xl border border-border bg-card p-3 text-left transition-all hover:-translate-y-px hover:border-primary/40 hover:shadow-sm disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:border-border disabled:hover:shadow-none",
                      comingSoon && "opacity-45",
                    )}
                  >
                    <span
                      className="grid size-8 place-items-center rounded-lg text-white transition-transform group-hover:scale-105"
                      style={{ backgroundColor: folder.hue }}
                    >
                      <Icon className="size-4" aria-hidden="true" />
                    </span>
                    <span className="text-[0.8rem] font-semibold">
                      {t(`folders.${folder.id}`)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {t("sidebar.itemsCount", { count: folderCount(folder.id, folder.count) })}
                    </span>
                  </button>
                )
              })}
            </div>
          </Section>
        </div>
      )}
    </aside>
  )
}

function TaskTracker() {
  const { chapters, progress } = useTaskTracker()
  const { t } = useLanguage()

  return (
    <Section title={t("sidebar.taskTracker")} className="flex min-h-0 flex-1 flex-col">
      {chapters.length === 0 ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/60 px-4 py-8 text-center">
          <p className="text-sm font-medium text-foreground">{t("sidebar.noProblemSet")}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground text-pretty">
            {t("sidebar.assignProblemSet")}
          </p>
        </div>
      ) : (
        <>
          <div className="shrink-0 rounded-xl border border-border bg-card p-3">
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="font-medium text-muted-foreground">
                {t("sidebar.sessionProgress")}
              </span>
              <span className="font-semibold text-primary">{progress}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="bg-brand-gradient h-full rounded-full transition-all duration-700 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <div className="mt-2 flex shrink-0 flex-wrap gap-x-3 gap-y-1 px-1 text-[0.65rem] text-muted-foreground">
            {(["completed", "review", "skipped"] as const).map((status) => (
              <span key={status} className="flex items-center gap-1">
                <span className={cn("size-2 rounded-full", STATUS_STYLES[status].dot)} />
                {t(`status.${status}`)}
              </span>
            ))}
          </div>

          <div className="scroll-slim mt-2 min-h-0 flex-1 space-y-2 overflow-y-auto pr-0.5">
            {chapters.map((chapter) => (
              <ChapterAccordion key={chapter.id} chapter={chapter} />
            ))}
          </div>
        </>
      )}
    </Section>
  )
}

function ChapterAccordion({ chapter }: { chapter: Chapter }) {
  const [open, setOpen] = useState(true)
  const { activeQuestionId, startQuestion, onStartQuestion, onQuestionStatus, removeModule, cycleStatus } =
    useTaskTracker()
  const { t } = useLanguage()

  const done = chapter.questions.filter((q) => q.status === "completed").length

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex items-center gap-0.5 pr-1">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5 text-left transition-colors hover:bg-sidebar-accent/40"
        >
          <ChevronDown
            className={cn(
              "size-4 shrink-0 text-muted-foreground transition-transform duration-300",
              !open && "-rotate-90",
            )}
            aria-hidden="true"
          />
          <span className="min-w-0 flex-1">
            {chapter.documentName ? (
              <span className="block truncate text-[0.65rem] text-muted-foreground">
                {chapter.documentName}
              </span>
            ) : null}
            <span className="block truncate text-[0.8rem] font-semibold">
              {chapter.title}
            </span>
            <span className="block text-xs text-muted-foreground">
              {t("sidebar.completedCount", {
                done,
                total: chapter.questions.length,
              })}
            </span>
          </span>
        </button>
        <button
          type="button"
          aria-label={t("sidebar.removeSection", { title: chapter.title })}
          onClick={(event) => {
            event.stopPropagation()
            removeModule(chapter.id)
          }}
          className="grid size-7 shrink-0 place-items-center rounded-lg text-muted-foreground/70 transition-colors hover:bg-destructive/10 hover:text-destructive"
        >
          <Trash2 className="size-3.5" aria-hidden="true" />
        </button>
      </div>

      {open && (
        <div className="grid grid-cols-5 gap-1.5 px-3 pb-3">
          {chapter.questions.map((q) => {
            const styles = STATUS_STYLES[q.status]
            const isActive = q.id === activeQuestionId
            return (
              <button
                key={q.id}
                type="button"
                onClick={(event) => {
                  if (event.altKey || event.metaKey || event.ctrlKey) {
                    event.preventDefault()
                    startQuestion(q.id)
                    onStartQuestion?.(q.id, true)
                    return
                  }
                  const next = nextQuestionStatus(q.status)
                  cycleStatus(q.id)
                  startQuestion(q.id)
                  onQuestionStatus?.(q.id, next)
                }}
                title={t("sidebar.questionTitle", {
                  label: q.label,
                  status: t(`status.${q.status}`),
                })}
                aria-label={t("sidebar.questionAria", {
                  label: q.label,
                  status: t(`status.${q.status}`),
                })}
                className={cn(
                  "grid h-8 place-items-center rounded-lg border text-xs font-semibold tabular-nums transition-all",
                  styles.pill,
                  isActive && "ring-2 ring-primary ring-offset-1 ring-offset-card",
                )}
              >
                {q.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

function Section({
  title,
  children,
  className,
}: {
  title: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={className}>
      <h2 className="mb-2 flex items-center gap-1.5 px-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {title}
      </h2>
      {children}
    </section>
  )
}

function CollapsedRail() {
  const { t } = useLanguage()
  const router = useRouter()
  return (
    <div className="flex flex-1 flex-col items-center gap-4 py-4">
      {MEMORY_FOLDERS.map((f) => {
        const Icon = f.icon
        const comingSoon = COMING_SOON_FOLDERS.has(f.id)
        const route = comingSoon ? undefined : FOLDER_ROUTES[f.id]
        return (
          <button
            key={f.id}
            type="button"
            disabled={!route}
            onClick={route ? () => router.push(route) : undefined}
            title={comingSoon ? t("collections.comingSoon") : t(`folders.${f.id}`)}
            aria-label={
              comingSoon
                ? `${t(`folders.${f.id}`)}. ${t("collections.comingSoon")}`
                : t(`folders.${f.id}`)
            }
            className={cn(
              "grid size-9 place-items-center rounded-xl text-white disabled:cursor-not-allowed",
              comingSoon && "opacity-45",
            )}
            style={{ backgroundColor: f.hue }}
          >
            <Icon className="size-4" aria-hidden="true" />
          </button>
        )
      })}
    </div>
  )
}
