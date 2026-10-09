"use client"

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react"
import {
  appendQuestionsToChapter,
  buildAssignedChapter,
  collapseChapters,
  sameChapterGroup,
  type Chapter,
  type Question,
  type QuestionStatus,
} from "./data"

/** Same three states as the chat Completed / Needs Review / Skip buttons. */
const STATUS_CYCLE: QuestionStatus[] = ["completed", "review", "skipped"]

export function nextQuestionStatus(status: QuestionStatus): QuestionStatus {
  if (status === "unanswered") return "completed"
  const idx = STATUS_CYCLE.indexOf(status)
  const safe = idx < 0 ? 0 : idx
  return STATUS_CYCLE[(safe + 1) % STATUS_CYCLE.length]
}

type ActiveQuestion = {
  chapterId: string
  chapterTitle: string
  question: Question
}

type AssignModuleInput = {
  title: string
  questionNumbers: number[]
  documentName?: string
}

type TaskTrackerValue = {
  chapters: Chapter[]
  /** Completion % based on green items, ignoring grey (skipped) items. */
  progress: number
  activeQuestionId: string | null
  activeQuestion: ActiveQuestion | null
  setActiveQuestionId: (id: string | null) => void
  /** Advance a question to its next state in the cycle. */
  cycleStatus: (questionId: string) => void
  /** Set a question directly to a specific state. */
  setStatus: (questionId: string, status: QuestionStatus) => void
  /** Keep the board card's problem text on the question that owns it. */
  rememberQuestionStem: (questionId: string, stem: string) => void
  /** Remember whether the automatic Review chip was already used for this explanation. */
  setAutoReviewed: (questionId: string, autoReviewed: boolean) => void
  assignModule: (input: AssignModuleInput) => string | null
  /** Remove a section and all of its question badges. */
  removeModule: (chapterId: string) => void
  clearModules: () => void
  hydrateTracker: (chapters: Chapter[], activeQuestionId: string | null) => void
  /** Activate the next unanswered question after `fromQuestionId`. */
  advanceToNextUnanswered: (fromQuestionId?: string) => void
  /** Mark a badge as the question being started (does not change status). */
  startQuestion: (questionId: string) => void
  /** ChatPane registers the JIT extract+solve launcher; sidebar calls it. */
  onStartQuestion: ((questionId: string, detailed?: boolean) => void) | null
  setOnStartQuestion: (
    handler: ((questionId: string, detailed?: boolean) => void) | null,
  ) => void
  /** ChatPane registers the local status listener used for the review chip. */
  onQuestionStatus: ((questionId: string, status: QuestionStatus) => void) | null
  setOnQuestionStatus: (
    handler: ((questionId: string, status: QuestionStatus) => void) | null,
  ) => void
}

const TaskTrackerContext = createContext<TaskTrackerValue | null>(null)

export function useTaskTracker() {
  const ctx = useContext(TaskTrackerContext)
  if (!ctx) {
    throw new Error("useTaskTracker must be used within a TaskTrackerProvider")
  }
  return ctx
}

function applyStatus(
  chapters: Chapter[],
  questionId: string,
  status: QuestionStatus,
): Chapter[] {
  return chapters.map((chapter) => ({
    ...chapter,
    questions: chapter.questions.map((q) =>
      q.id === questionId ? { ...q, status } : q,
    ),
  }))
}

export function nextUnansweredId(chapters: Chapter[], fromId?: string | null): string | null {
  const all = chapters.flatMap((c) => c.questions)
  if (all.length === 0) return null
  const start = fromId ? all.findIndex((q) => q.id === fromId) : -1
  const rest = start >= 0 ? [...all.slice(start + 1), ...all.slice(0, start)] : all
  const found = rest.find((q) => q.status === "unanswered")
  return found?.id ?? fromId ?? all[0]?.id ?? null
}

export function TaskTrackerProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [chapters, setChapters] = useState<Chapter[]>([])
  const [activeQuestionId, setActiveQuestionId] = useState<string | null>(null)
  const [onStartQuestion, setOnStartQuestionState] = useState<
    ((questionId: string, detailed?: boolean) => void) | null
  >(null)
  const [onQuestionStatus, setOnQuestionStatusState] = useState<
    ((questionId: string, status: QuestionStatus) => void) | null
  >(null)

  const setOnStartQuestion = useCallback(
    (handler: ((questionId: string, detailed?: boolean) => void) | null) => {
      setOnStartQuestionState(() => handler)
    },
    [],
  )

  const setOnQuestionStatus = useCallback(
    (handler: ((questionId: string, status: QuestionStatus) => void) | null) => {
      setOnQuestionStatusState(() => handler)
    },
    [],
  )

  const setStatus = useCallback((questionId: string, status: QuestionStatus) => {
    setChapters((prev) => applyStatus(prev, questionId, status))
  }, [])

  const rememberQuestionStem = useCallback((questionId: string, stem: string) => {
    const nextStem = stem.trim().slice(0, 4000)
    if (!nextStem) return
    setChapters((prev) => {
      let changed = false
      const next = prev.map((chapter) => ({
        ...chapter,
        questions: chapter.questions.map((question) => {
          if (question.id !== questionId || question.stem === nextStem) return question
          changed = true
          return { ...question, stem: nextStem }
        }),
      }))
      return changed ? next : prev
    })
  }, [])

  const setAutoReviewed = useCallback((questionId: string, autoReviewed: boolean) => {
    setChapters((prev) => {
      let changed = false
      const next = prev.map((chapter) => ({
        ...chapter,
        questions: chapter.questions.map((question) => {
          if (question.id !== questionId || Boolean(question.autoReviewed) === autoReviewed) {
            return question
          }
          changed = true
          return { ...question, autoReviewed }
        }),
      }))
      return changed ? next : prev
    })
  }, [])

  const cycleStatus = useCallback((questionId: string) => {
    setChapters((prev) =>
      prev.map((chapter) => ({
        ...chapter,
        questions: chapter.questions.map((q) => {
          if (q.id !== questionId) return q
          return { ...q, status: nextQuestionStatus(q.status) }
        }),
      })),
    )
  }, [])

  const assignModule = useCallback((input: AssignModuleInput) => {
    const numbers = [...new Set(input.questionNumbers.filter((n) => n > 0))]
    if (numbers.length === 0) return null
    let firstNewId: string | null = null
    setChapters((prev) => {
      const collapsed = collapseChapters(prev)
      const probe: Chapter = {
        id: "",
        title: input.title,
        documentName: input.documentName || undefined,
        questions: [],
      }
      const existing = collapsed.find((chapter) => sameChapterGroup(chapter, probe))
      if (!existing) {
        const chapter = buildAssignedChapter(input.title, numbers, input.documentName ?? "")
        firstNewId = chapter.questions[0]?.id ?? null
        return [...collapsed, chapter]
      }
      const before = new Set(existing.questions.map((q) => q.id))
      let merged = appendQuestionsToChapter(existing, numbers)
      if (!merged.documentName && input.documentName) {
        merged = { ...merged, documentName: input.documentName }
      }
      const added = merged.questions.find((q) => !before.has(q.id))
      firstNewId = added?.id ?? null
      if (merged === existing && collapsed === prev) return prev
      return collapsed.map((chapter) => (chapter.id === existing.id ? merged : chapter))
    })
    return firstNewId
  }, [])

  const removeModule = useCallback((chapterId: string) => {
    setChapters((prev) => {
      const next = prev.filter((chapter) => chapter.id !== chapterId)
      setActiveQuestionId((id) => {
        if (!id) return null
        const stillThere = next.some((chapter) => chapter.questions.some((q) => q.id === id))
        return stillThere ? id : null
      })
      return next
    })
  }, [])

  const startQuestion = useCallback((questionId: string) => {
    setActiveQuestionId(questionId)
  }, [])

  const clearModules = useCallback(() => {
    setChapters([])
    setActiveQuestionId(null)
  }, [])

  const hydrateTracker = useCallback(
    (nextChapters: Chapter[], nextActive: string | null) => {
      setChapters(collapseChapters(nextChapters))
      setActiveQuestionId(nextActive)
    },
    [],
  )

  const advanceToNextUnanswered = useCallback((fromQuestionId?: string) => {
    setChapters((prev) => {
      const nextId = nextUnansweredId(prev, fromQuestionId)
      setActiveQuestionId(nextId)
      return prev
    })
  }, [])

  const progress = useMemo(() => {
    const all = chapters.flatMap((c) => c.questions)
    const counted = all.filter((q) => q.status !== "skipped")
    if (counted.length === 0) return 0
    const completed = counted.filter((q) => q.status === "completed").length
    return Math.round((completed / counted.length) * 100)
  }, [chapters])

  const activeQuestion = useMemo<ActiveQuestion | null>(() => {
    if (!activeQuestionId) return null
    for (const chapter of chapters) {
      const question = chapter.questions.find((q) => q.id === activeQuestionId)
      if (question) {
        return { chapterId: chapter.id, chapterTitle: chapter.title, question }
      }
    }
    return null
  }, [activeQuestionId, chapters])

  const value = useMemo<TaskTrackerValue>(
    () => ({
      chapters,
      progress,
      activeQuestionId,
      activeQuestion,
      setActiveQuestionId,
      cycleStatus,
      setStatus,
      rememberQuestionStem,
      setAutoReviewed,
      assignModule,
      removeModule,
      clearModules,
      hydrateTracker,
      advanceToNextUnanswered,
      startQuestion,
      onStartQuestion,
      setOnStartQuestion,
      onQuestionStatus,
      setOnQuestionStatus,
    }),
    [
      chapters,
      progress,
      activeQuestionId,
      activeQuestion,
      cycleStatus,
      setStatus,
      rememberQuestionStem,
      setAutoReviewed,
      assignModule,
      removeModule,
      clearModules,
      hydrateTracker,
      advanceToNextUnanswered,
      startQuestion,
      onStartQuestion,
      setOnStartQuestion,
      onQuestionStatus,
      setOnQuestionStatus,
    ],
  )

  return (
    <TaskTrackerContext.Provider value={value}>
      {children}
    </TaskTrackerContext.Provider>
  )
}

/** Shared visual tokens for each question status. */
export const STATUS_STYLES: Record<
  QuestionStatus,
  { pill: string; dot: string; label: string }
> = {
  unanswered: {
    pill: "border-border bg-card text-muted-foreground hover:border-primary/50 hover:text-foreground",
    dot: "bg-muted-foreground/40",
    label: "Unanswered",
  },
  completed: {
    pill: "border-transparent bg-emerald-500 text-white shadow-sm",
    dot: "bg-emerald-500",
    label: "Completed",
  },
  review: {
    pill: "border-transparent bg-red-500 text-white shadow-sm",
    dot: "bg-red-500",
    label: "Needs Review",
  },
  skipped: {
    pill: "border-transparent bg-muted-foreground/70 text-white",
    dot: "bg-muted-foreground/70",
    label: "Skipped",
  },
}
