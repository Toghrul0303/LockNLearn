"use client"

import { useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { AlertCircle, ArrowLeft, BookMarked, FunctionSquare, LineChart, NotebookPen, Sparkles, Trash2, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  isDeskBookmark,
  isQuestionBookmark,
  isSessionBookmark,
  useMemoryBox,
  type QuestionBookmark,
  type SavedBookmark,
  type SavedFormula,
  type SavedGraph,
  type SavedStruggle,
} from "@/components/workspace/memory-box-context"
import { rememberLastThread } from "@/components/workspace/session-context"
import { DynamicDeskWidget } from "@/components/workspace/dynamic-desk-widget"
import { MathMarkdown } from "@/components/workspace/math-markdown"
import { mathRehypePlugins, mathRemarkPlugins } from "@/lib/markdown-math"
import { useLanguage } from "@/components/workspace/language-context"
import { localeToBcp47 } from "@/lib/i18n"
import { QuestionPreviewModal } from "@/components/collections/question-preview-modal"

type TabId = "formulas" | "bookmarks"

function SavedMath({ source }: { source: string }) {
  return (
    <div className="overflow-x-auto rounded-xl bg-secondary/70 px-3 py-2 text-sm">
      <MathMarkdown remarkPlugins={mathRemarkPlugins} rehypePlugins={mathRehypePlugins}>
        {source}
      </MathMarkdown>
    </div>
  )
}

function formatDate(ts: number, locale: string) {
  return new Date(ts).toLocaleString(locale, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

/** Dashboard at `/collections` — everything saved via the chat's
 * "Save to FormulaBox" / "Save to GraphBox" buttons lives here, grouped
 * into two tabs. Reads the initial tab from `?tab=formulas|graphs`. */
export function CollectionsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const rawTab = searchParams.get("tab")
  const initialTab: TabId = rawTab === "bookmarks" ? "bookmarks" : "formulas"
  const [tab, setTab] = useState<TabId>(initialTab)
  const [preview, setPreview] = useState<QuestionBookmark | null>(null)
  const {
    savedFormulas,
    savedBookmarks,
    removeFormula,
    removeBookmark,
  } = useMemoryBox()
  const { locale, t } = useLanguage()
  const dateLocale = localeToBcp47(locale)

  return (
    <div className="scroll-slim min-h-dvh overflow-y-auto bg-background text-foreground">
      <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-border bg-background/90 px-6 py-4 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <span className="bg-brand-gradient ring-brand-glow grid size-9 place-items-center rounded-xl text-white">
            <Sparkles className="size-4" aria-hidden="true" />
          </span>
          <div>
            <h1 className="font-display text-lg font-semibold">{t("collections.title")}</h1>
            <p className="text-xs text-muted-foreground">
              {t("collections.subtitle")}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => router.push("/workspace")}
          className="bg-brand-gradient ring-brand-glow flex shrink-0 items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-white transition-transform hover:-translate-y-px"
        >
          <ArrowLeft className="size-4" aria-hidden="true" />
          {t("collections.backToDesk")}
        </button>
      </header>

      <div className="mx-auto max-w-5xl px-6 py-6">
        <div className="mb-6 inline-flex rounded-xl border border-border bg-card p-1">
          <TabButton
            active={tab === "formulas"}
            onClick={() => setTab("formulas")}
            icon={FunctionSquare}
            label={t("collections.savedFormulas")}
            count={savedFormulas.length}
          />
          <TabButton
            active={tab === "bookmarks"}
            onClick={() => setTab("bookmarks")}
            icon={BookMarked}
            label={t("collections.bookmarks")}
            count={savedBookmarks.length}
          />
          <TabButton
            disabled
            title={t("collections.comingSoon")}
            icon={LineChart}
            label={t("collections.savedGraphs")}
          />
          <TabButton
            disabled
            title={t("collections.comingSoon")}
            icon={NotebookPen}
            label={t("folders.summaries")}
          />
        </div>

        {tab === "bookmarks" ? (
          <BookmarksGrid
            bookmarks={savedBookmarks}
            onRemove={removeBookmark}
            dateLocale={dateLocale}
            onOpenQuestion={setPreview}
          />
        ) : (
          <FormulasGrid formulas={savedFormulas} onRemove={removeFormula} dateLocale={dateLocale} />
        )}
      </div>
      {preview ? <QuestionPreviewModal bookmark={preview} onClose={() => setPreview(null)} /> : null}
    </div>
  )
}

function TabButton({
  active = false,
  onClick,
  icon: Icon,
  label,
  count,
  disabled = false,
  title,
}: {
  active?: boolean
  onClick?: () => void
  icon: LucideIcon
  label: string
  count?: number
  disabled?: boolean
  title?: string
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      title={title}
      aria-pressed={disabled ? undefined : active}
      className={cn(
        "flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-semibold transition-colors",
        disabled
          ? "cursor-not-allowed text-muted-foreground/50 opacity-50"
          : active
            ? "bg-brand-gradient text-white shadow-sm"
            : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
      )}
    >
      <Icon className="size-4" aria-hidden="true" />
      {label}
      {count != null ? (
        <span
          className={cn(
            "grid min-w-5 place-items-center rounded-full px-1.5 py-0.5 text-[0.7rem] font-bold tabular-nums",
            active ? "bg-white/20" : "bg-secondary text-muted-foreground",
          )}
        >
          {count}
        </span>
      ) : null}
    </button>
  )
}

function EmptyState({ icon: Icon, label }: { icon: LucideIcon; label: string }) {
  return (
    <div className="grid min-h-[220px] place-items-center rounded-2xl border border-dashed border-border/70 text-center">
      <div className="flex flex-col items-center gap-2 px-6">
        <Icon className="size-6 text-muted-foreground/50" aria-hidden="true" />
        <p className="text-sm text-muted-foreground text-pretty">{label}</p>
      </div>
    </div>
  )
}

function FormulasGrid({
  formulas,
  onRemove,
  dateLocale,
}: {
  formulas: SavedFormula[]
  onRemove: (id: string) => void
  dateLocale: string
}) {
  const { t } = useLanguage()
  if (formulas.length === 0) {
    return (
      <EmptyState
        icon={FunctionSquare}
        label={t("collections.emptyFormulas")}
      />
    )
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {formulas.map((f) => (
        <div
          key={f.id}
          className="group relative flex flex-col gap-2 rounded-2xl border border-border bg-card p-4 shadow-sm"
        >
          <button
            type="button"
            onClick={() => onRemove(f.id)}
            aria-label={t("collections.removeFormula")}
            className="absolute right-3 top-3 grid size-7 place-items-center rounded-lg text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
          >
            <Trash2 className="size-3.5" aria-hidden="true" />
          </button>

          {f.context ? <p className="pr-8 text-sm font-semibold">{f.context}</p> : null}
          <SavedMath source={f.formula} />
          <span className="text-xs font-medium text-muted-foreground">
            {formatDate(f.savedAt, dateLocale)}
          </span>
        </div>
      ))}
    </div>
  )
}

function GraphsGrid({
  graphs,
  onRemove,
  dateLocale,
}: {
  graphs: SavedGraph[]
  onRemove: (id: string) => void
  dateLocale: string
}) {
  const { t } = useLanguage()
  if (graphs.length === 0) {
    return (
      <EmptyState
        icon={LineChart}
        label={t("collections.emptyGraphs")}
      />
    )
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {graphs.map((g) => (
        <div
          key={g.id}
          className="group relative flex flex-col gap-2 rounded-2xl border border-border bg-card p-3 shadow-sm"
        >
          <button
            type="button"
            onClick={() => onRemove(g.id)}
            aria-label={t("collections.removeGraph")}
            className="absolute right-3 top-3 z-10 grid size-7 place-items-center rounded-lg bg-popover/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur-sm transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
          >
            <Trash2 className="size-3.5" aria-hidden="true" />
          </button>

          <div className="h-64 w-full">
            <DynamicDeskWidget update={g.chart} />
          </div>

          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-medium text-muted-foreground">{formatDate(g.savedAt, dateLocale)}</span>
          </div>
          {g.context && (
            <p className="line-clamp-2 px-1 text-sm leading-relaxed text-muted-foreground text-pretty">
              {g.context}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}

function BookmarksGrid({
  bookmarks,
  onRemove,
  dateLocale,
  onOpenQuestion,
}: {
  bookmarks: SavedBookmark[]
  onRemove: (id: string) => void
  dateLocale: string
  onOpenQuestion: (bookmark: QuestionBookmark) => void
}) {
  const router = useRouter()
  const { t } = useLanguage()
  const desk = bookmarks.filter(isDeskBookmark)
  const sessions = bookmarks.filter(isSessionBookmark)
  const questions = bookmarks.filter(isQuestionBookmark)

  if (bookmarks.length === 0) {
    return (
      <EmptyState
        icon={BookMarked}
        label={t("collections.emptyBookmarks")}
      />
    )
  }

  return (
    <div className="flex flex-col gap-6">
      {sessions.length > 0 ? (
        <div className="flex flex-col gap-2">
          {sessions.map((bookmark) => (
            <div
              key={bookmark.id}
              className="group relative flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 shadow-sm"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold">{bookmark.title}</p>
                <p className="text-xs font-medium text-muted-foreground">
                  {formatDate(bookmark.savedAt, dateLocale)}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  rememberLastThread(bookmark.sessionId)
                  router.push("/workspace")
                }}
                className="bg-brand-gradient shrink-0 rounded-lg px-3 py-1.5 text-xs font-semibold text-white"
              >
                {t("collections.open")}
              </button>
              <button
                type="button"
                onClick={() => onRemove(bookmark.id)}
                aria-label={t("collections.removeSessionBookmark")}
                className="grid size-7 shrink-0 place-items-center rounded-lg text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {questions.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {questions.map((bookmark) => (
            <div
              key={bookmark.id}
              role="button"
              tabIndex={0}
              onClick={() => onOpenQuestion(bookmark)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onOpenQuestion(bookmark)
                }
              }}
              className="group relative flex cursor-pointer flex-col gap-2 rounded-2xl border border-border bg-card p-4 shadow-sm"
            >
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  onRemove(bookmark.id)
                }}
                aria-label={t("collections.removeBookmark")}
                className="absolute right-3 top-3 z-10 grid size-7 place-items-center rounded-lg bg-popover/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur-sm transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
              </button>
              <p className="pr-8 text-sm font-semibold">{bookmark.title}</p>
              {bookmark.prompt ? <SavedMath source={bookmark.prompt} /> : null}
              <span className="text-xs font-medium text-muted-foreground">
                {formatDate(bookmark.savedAt, dateLocale)}
              </span>
            </div>
          ))}
        </div>
      ) : null}

      {desk.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {desk.map((bookmark) => (
            <div
              key={bookmark.id}
              className="group relative flex flex-col gap-2 rounded-2xl border border-border bg-card p-3 shadow-sm"
            >
              <button
                type="button"
                onClick={() => onRemove(bookmark.id)}
                aria-label={t("collections.removeBookmark")}
                className="absolute right-3 top-3 z-10 grid size-7 place-items-center rounded-lg bg-popover/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur-sm transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
              </button>
              <div className="h-64 w-full">
                <DynamicDeskWidget update={bookmark.item} />
              </div>
              <div className="flex items-center justify-between px-1">
                <span className="text-xs font-medium text-muted-foreground">
                  {formatDate(bookmark.savedAt, dateLocale)}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function StrugglesGrid({
  struggles,
  onRemove,
  dateLocale,
}: {
  struggles: SavedStruggle[]
  onRemove: (id: string) => void
  dateLocale: string
}) {
  const { t } = useLanguage()
  if (struggles.length === 0) {
    return (
      <EmptyState
        icon={AlertCircle}
        label={t("collections.emptyReview")}
      />
    )
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {struggles.map((s) => (
        <div
          key={s.id}
          className="group relative flex flex-col gap-3 rounded-2xl border border-border bg-card p-4 shadow-sm"
        >
          <button
            type="button"
            onClick={() => onRemove(s.id)}
            aria-label={t("collections.removeReview")}
            className="absolute right-3 top-3 grid size-7 place-items-center rounded-lg text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
          >
            <Trash2 className="size-3.5" aria-hidden="true" />
          </button>
          <div className="flex items-center gap-2">
            <span className="grid size-8 place-items-center rounded-lg bg-red-500 text-white">
              <AlertCircle className="size-4" aria-hidden="true" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{s.topic}</p>
              <p className="text-xs font-medium text-muted-foreground">{formatDate(s.savedAt, dateLocale)}</p>
            </div>
          </div>
          {s.excerpt ? (
            <p className="line-clamp-5 text-sm leading-relaxed text-muted-foreground text-pretty">
              {s.excerpt}
            </p>
          ) : null}
        </div>
      ))}
    </div>
  )
}
