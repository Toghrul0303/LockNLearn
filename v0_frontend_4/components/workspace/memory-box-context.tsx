"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import type { ChartDeskUpdate, DeskItem } from "./desk-context"
import { useLanguage } from "./language-context"
import { useAuth } from "./auth-context"
import {
  deleteRemoteRow,
  fetchRemoteMemory,
  mergeById,
  pushMergedMemory,
  upsertBookmark,
  upsertFormula,
  upsertGraph,
  upsertStruggle,
} from "@/lib/memory-box-sync"
import { BOOKMARKS_KEY, FORMULAS_KEY, GRAPHS_KEY, STRUGGLES_KEY } from "@/lib/user-data"

export type SavedFormula = {
  id: string
  formula: string
  /** The chat message/question this formula came from, if any. */
  context?: string
  savedAt: number
}

export type SavedGraph = {
  id: string
  chart: ChartDeskUpdate
  context?: string
  savedAt: number
}

export type SavedStruggle = {
  id: string
  topic: string
  excerpt: string
  questionId?: string
  savedAt: number
}

export type DeskBookmark = {
  id: string
  kind?: "desk"
  item: DeskItem
  savedAt: number
}

export type SessionBookmark = {
  id: string
  kind: "session"
  sessionId: string
  title: string
  savedAt: number
}

export type QuestionClusterShape = {
  id: string
  type: string
  x: number
  y: number
  parentId: string
  props: Record<string, unknown>
}

export type QuestionBookmark = {
  id: string
  kind: "question"
  questionId: string
  title: string
  prompt: string
  cluster: QuestionClusterShape[]
  savedAt: number
}

export type SavedBookmark = DeskBookmark | SessionBookmark | QuestionBookmark

export function isSessionBookmark(bookmark: SavedBookmark): bookmark is SessionBookmark {
  return bookmark.kind === "session"
}

export function isDeskBookmark(bookmark: SavedBookmark): bookmark is DeskBookmark {
  return bookmark.kind !== "session" && bookmark.kind !== "question" && "item" in bookmark && Boolean(bookmark.item)
}

export function isQuestionBookmark(bookmark: SavedBookmark): bookmark is QuestionBookmark {
  return bookmark.kind === "question"
}

const TOAST_DURATION_MS = 2600

function loadFromStorage<T>(key: string): T[] {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T[]) : []
  } catch {
    return []
  }
}

function saveToStorage<T>(key: string, items: T[]) {
  try {
    window.localStorage.setItem(key, JSON.stringify(items))
  } catch {
    /* localStorage may be unavailable (private browsing, quota, etc.) — saves
     * still work for the current session via React state either way. */
  }
}

type MemoryBoxContextValue = {
  savedFormulas: SavedFormula[]
  savedGraphs: SavedGraph[]
  savedStruggles: SavedStruggle[]
  savedBookmarks: SavedBookmark[]
  saveFormula: (formula: string, context?: string) => string
  renameFormula: (id: string, context: string) => void
  saveGraph: (chart: ChartDeskUpdate, context?: string) => void
  saveStruggle: (topic: string, excerpt: string, questionId?: string) => void
  saveBookmark: (item: DeskItem) => boolean
  saveQuestionSnapshot: (snapshot: {
    questionId: string
    title: string
    prompt: string
    cluster: QuestionClusterShape[]
  }) => boolean
  removeFormula: (id: string) => void
  removeGraph: (id: string) => void
  removeStruggle: (id: string) => void
  removeBookmark: (id: string) => void
  /** Currently visible toast message, or null. */
  toastMessage: string | null
}

const MemoryBoxContext = createContext<MemoryBoxContextValue>({
  savedFormulas: [],
  savedGraphs: [],
  savedStruggles: [],
  savedBookmarks: [],
  saveFormula: () => "",
  renameFormula: () => {},
  saveGraph: () => {},
  saveStruggle: () => {},
  saveBookmark: () => false,
  saveQuestionSnapshot: () => false,
  removeFormula: () => {},
  removeGraph: () => {},
  removeStruggle: () => {},
  removeBookmark: () => {},
  toastMessage: null,
})

export function MemoryBoxProvider({ children }: { children: ReactNode }) {
  const { t } = useLanguage()
  const { user } = useAuth()
  const userId = user?.id ?? null
  const userIdRef = useRef(userId)
  userIdRef.current = userId
  const formulasRef = useRef<SavedFormula[]>([])
  const [localReady, setLocalReady] = useState(false)
  const [savedFormulas, setSavedFormulas] = useState<SavedFormula[]>([])
  const [savedGraphs, setSavedGraphs] = useState<SavedGraph[]>([])
  const [savedStruggles, setSavedStruggles] = useState<SavedStruggle[]>([])
  const [savedBookmarks, setSavedBookmarks] = useState<SavedBookmark[]>([])
  const [toastMessage, setToastMessage] = useState<string | null>(null)
  formulasRef.current = savedFormulas

  // Loaded lazily on mount (not lazy useState initializer) so the very same
  // provider works identically whether rendered on `/` or `/collections`.
  useEffect(() => {
    setSavedFormulas(loadFromStorage<SavedFormula>(FORMULAS_KEY))
    setSavedGraphs(loadFromStorage<SavedGraph>(GRAPHS_KEY))
    setSavedStruggles(loadFromStorage<SavedStruggle>(STRUGGLES_KEY))
    setSavedBookmarks(loadFromStorage<SavedBookmark>(BOOKMARKS_KEY))
    setLocalReady(true)
  }, [])

  useEffect(() => {
    if (!localReady || !userId) return
    let cancelled = false
    void (async () => {
      const remote = await fetchRemoteMemory(userId)
      if (cancelled || !remote) return
      const formulas = mergeById(loadFromStorage<SavedFormula>(FORMULAS_KEY), remote.formulas)
      const graphs = mergeById(loadFromStorage<SavedGraph>(GRAPHS_KEY), remote.graphs)
      const struggles = mergeById(loadFromStorage<SavedStruggle>(STRUGGLES_KEY), remote.struggles)
      const bookmarks = mergeById(loadFromStorage<SavedBookmark>(BOOKMARKS_KEY), remote.bookmarks)
      if (cancelled) return
      setSavedFormulas(formulas)
      setSavedGraphs(graphs)
      setSavedStruggles(struggles)
      setSavedBookmarks(bookmarks)
      saveToStorage(FORMULAS_KEY, formulas)
      saveToStorage(GRAPHS_KEY, graphs)
      saveToStorage(STRUGGLES_KEY, struggles)
      saveToStorage(BOOKMARKS_KEY, bookmarks)
      void pushMergedMemory(userId, { formulas, graphs, struggles, bookmarks })
    })()
    return () => {
      cancelled = true
    }
  }, [localReady, userId])

  const showToast = useCallback((message: string) => {
    setToastMessage(message)
    window.setTimeout(() => {
      setToastMessage((current) => (current === message ? null : current))
    }, TOAST_DURATION_MS)
  }, [])

  const saveFormula = useCallback(
    (formula: string, context?: string) => {
      const item: SavedFormula = { id: crypto.randomUUID(), formula, context, savedAt: Date.now() }
      setSavedFormulas((prev) => {
        const next = [item, ...prev]
        formulasRef.current = next
        saveToStorage(FORMULAS_KEY, next)
        return next
      })
      const uid = userIdRef.current
      if (uid) void upsertFormula(uid, item)
      showToast(t("toast.savedMemory"))
      return item.id
    },
    [showToast, t],
  )

  const renameFormula = useCallback((id: string, context: string) => {
    const current = formulasRef.current.find((item) => item.id === id)
    if (!current) return
    const updated = { ...current, context }
    setSavedFormulas((prev) => {
      const next = prev.map((item) => (item.id === id ? updated : item))
      formulasRef.current = next
      saveToStorage(FORMULAS_KEY, next)
      return next
    })
    const uid = userIdRef.current
    if (uid) void upsertFormula(uid, updated)
  }, [])

  const saveGraph = useCallback(
    (chart: ChartDeskUpdate, context?: string) => {
      const item: SavedGraph = { id: crypto.randomUUID(), chart, context, savedAt: Date.now() }
      setSavedGraphs((prev) => {
        const next = [item, ...prev]
        saveToStorage(GRAPHS_KEY, next)
        return next
      })
      const uid = userIdRef.current
      if (uid) void upsertGraph(uid, item)
      showToast(t("toast.savedMemory"))
    },
    [showToast, t],
  )

  const removeFormula = useCallback((id: string) => {
    setSavedFormulas((prev) => {
      const next = prev.filter((f) => f.id !== id)
      saveToStorage(FORMULAS_KEY, next)
      return next
    })
    const uid = userIdRef.current
    if (uid) void deleteRemoteRow("memory_formulas", uid, id)
  }, [])

  const removeGraph = useCallback((id: string) => {
    setSavedGraphs((prev) => {
      const next = prev.filter((g) => g.id !== id)
      saveToStorage(GRAPHS_KEY, next)
      return next
    })
    const uid = userIdRef.current
    if (uid) void deleteRemoteRow("memory_graphs", uid, id)
  }, [])

  const saveBookmark = useCallback(
    (item: DeskItem): boolean => {
      let added = false
      let created: SavedBookmark | null = null
      setSavedBookmarks((prev) => {
        if (prev.some((bookmark) => isDeskBookmark(bookmark) && bookmark.item.id === item.id)) {
          return prev
        }
        added = true
        created = {
          id: crypto.randomUUID(),
          kind: "desk",
          item: { ...item, bookmarked: true },
          savedAt: Date.now(),
        }
        const next: SavedBookmark[] = [created, ...prev]
        saveToStorage(BOOKMARKS_KEY, next)
        return next
      })
      const uid = userIdRef.current
      if (added && created && uid) void upsertBookmark(uid, created)
      showToast(added ? t("toast.savedBookmarks") : t("toast.alreadyBookmarks"))
      return added
    },
    [showToast, t],
  )

  const saveQuestionSnapshot = useCallback(
    (snapshot: {
      questionId: string
      title: string
      prompt: string
      cluster: QuestionClusterShape[]
    }): boolean => {
      let wrote = false
      let saved: QuestionBookmark | null = null
      setSavedBookmarks((prev) => {
        const index = prev.findIndex(
          (bookmark) => isQuestionBookmark(bookmark) && bookmark.questionId === snapshot.questionId,
        )
        if (index >= 0) {
          const existing = prev[index]
          if (!isQuestionBookmark(existing)) return prev
          const updated: QuestionBookmark = {
            ...existing,
            title: snapshot.title,
            prompt: snapshot.prompt,
            cluster: snapshot.cluster,
            savedAt: Date.now(),
          }
          wrote = true
          saved = updated
          const next = prev.map((bookmark, i) => (i === index ? updated : bookmark))
          saveToStorage(BOOKMARKS_KEY, next)
          return next
        }
        const created: QuestionBookmark = {
          id: crypto.randomUUID(),
          kind: "question",
          questionId: snapshot.questionId,
          title: snapshot.title,
          prompt: snapshot.prompt,
          cluster: snapshot.cluster,
          savedAt: Date.now(),
        }
        wrote = true
        saved = created
        const next: SavedBookmark[] = [created, ...prev]
        saveToStorage(BOOKMARKS_KEY, next)
        return next
      })
      const uid = userIdRef.current
      if (wrote && saved && uid) void upsertBookmark(uid, saved)
      showToast(t("toast.savedQuestion"))
      return wrote
    },
    [showToast, t],
  )

  const removeBookmark = useCallback((id: string) => {
    setSavedBookmarks((prev) => {
      const next = prev.filter((bookmark) => bookmark.id !== id)
      saveToStorage(BOOKMARKS_KEY, next)
      return next
    })
    const uid = userIdRef.current
    if (uid) void deleteRemoteRow("memory_bookmarks", uid, id)
  }, [])

  const saveStruggle = useCallback(
    (topic: string, excerpt: string, questionId?: string) => {
      const item: SavedStruggle = {
        id: crypto.randomUUID(),
        topic,
        excerpt,
        questionId,
        savedAt: Date.now(),
      }
      setSavedStruggles((prev) => {
        const next = [item, ...prev]
        saveToStorage(STRUGGLES_KEY, next)
        return next
      })
      const uid = userIdRef.current
      if (uid) void upsertStruggle(uid, item)
      showToast(t("toast.savedReview"))
    },
    [showToast, t],
  )

  const removeStruggle = useCallback((id: string) => {
    setSavedStruggles((prev) => {
      const next = prev.filter((s) => s.id !== id)
      saveToStorage(STRUGGLES_KEY, next)
      return next
    })
    const uid = userIdRef.current
    if (uid) void deleteRemoteRow("memory_struggles", uid, id)
  }, [])

  const value = useMemo(
    () => ({
      savedFormulas,
      savedGraphs,
      savedStruggles,
      savedBookmarks,
      saveFormula,
      renameFormula,
      saveGraph,
      saveStruggle,
      saveBookmark,
      saveQuestionSnapshot,
      removeFormula,
      removeGraph,
      removeStruggle,
      removeBookmark,
      toastMessage,
    }),
    [
      savedFormulas,
      savedGraphs,
      savedStruggles,
      savedBookmarks,
      saveFormula,
      renameFormula,
      saveGraph,
      saveStruggle,
      saveBookmark,
      saveQuestionSnapshot,
      removeFormula,
      removeGraph,
      removeStruggle,
      removeBookmark,
      toastMessage,
    ],
  )

  return <MemoryBoxContext.Provider value={value}>{children}</MemoryBoxContext.Provider>
}

export function useMemoryBox() {
  return useContext(MemoryBoxContext)
}
