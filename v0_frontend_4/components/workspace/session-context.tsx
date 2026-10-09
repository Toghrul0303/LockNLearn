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
import type { Chapter, ChatMessage } from "./data"
import type { ActiveProblem, DeskItem, WhiteboardMeta } from "./desk-context"
import { deleteSessionBoards } from "@/lib/board-idb"
import {
  deleteRemoteSession,
  fetchRemoteSessions,
  mergeSessions,
  upsertSession,
} from "@/lib/session-sync"
import { SESSIONS_KEY as STORAGE_KEY } from "@/lib/user-data"
import { useAuth } from "./auth-context"

export const MAX_SESSIONS = 12

export type SessionRecord = {
  id: string
  title: string
  createdAt: number
  lastActiveAt: number
  messages: ChatMessage[]
  chapters: Chapter[]
  activeQuestionId: string | null
  deskItems: DeskItem[]
  activeProblem: ActiveProblem | null
  whiteboards?: WhiteboardMeta[]
  activeWhiteboardId?: string | null
  /** Supabase Storage path for the signed-in PDF, when one has been uploaded. */
  documentPath?: string | null
}

type StoredSessions = {
  sessions: SessionRecord[]
  lastThreadId?: string
}

type SessionContextValue = {
  /** Sent as `thread_id` in every `/submit_stream` request. */
  threadId: string
  sessions: SessionRecord[]
  isDraft: boolean
  prepareDraft: () => void
  commitSessionTitle: (firstPrompt: string, attachmentName?: string) => void
  selectSession: (id: string) => void
  getSession: (id: string) => SessionRecord | undefined
  patchSession: (id: string, patch: Partial<SessionRecord>) => void
  deleteSession: (id: string) => void
  /** Starts a brand-new draft session (empty chat, empty Desk). */
  resetSession: () => void
  /** Local load finished, and a signed-in cloud merge has settled. */
  sessionsReady: boolean
}

const SessionContext = createContext<SessionContextValue>({
  threadId: "default_session",
  sessions: [],
  isDraft: true,
  prepareDraft: () => {},
  commitSessionTitle: () => {},
  selectSession: () => {},
  getSession: () => undefined,
  patchSession: () => {},
  deleteSession: () => {},
  resetSession: () => {},
  sessionsReady: false,
})

function mintThreadId() {
  return `session_${Date.now()}`
}

function loadStored(): StoredSessions {
  if (typeof window === "undefined") return { sessions: [] }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { sessions: [] }
    const parsed = JSON.parse(raw) as StoredSessions
    if (!Array.isArray(parsed.sessions)) return { sessions: [] }
    return parsed
  } catch {
    return { sessions: [] }
  }
}

function saveStored(state: StoredSessions) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    /* quota / private browsing — in-memory sessions still work */
  }
}

export function titleFromFirstPrompt(prompt: string, attachmentName?: string): string {
  const trimmed = prompt.trim()
  if (/\[TASK START\]/i.test(trimmed)) {
    const parsed = trimmed.match(/question\s+(\d+)\s+from\s+"([^"]+)"/i)
    if (parsed) return `${parsed[2]}, sual ${parsed[1]}`
  } else {
    const words = trimmed.split(/\s+/).filter(Boolean)
    if (words.length > 0) {
      return words.length <= 6 ? words.join(" ") : `${words.slice(0, 6).join(" ")}…`
    }
  }
  if (attachmentName) {
    const stem = attachmentName.replace(/\.[^.]+$/, "").trim()
    return stem || attachmentName
  }
  return "New session"
}

function evictIfNeeded(
  sessions: SessionRecord[],
  keepId: string,
  onDrop?: (session: SessionRecord) => void,
): SessionRecord[] {
  if (sessions.length <= MAX_SESSIONS) return sessions
  const overflow = sessions.length - MAX_SESSIONS
  const byLru = [...sessions]
    .filter((s) => s.id !== keepId)
    .sort((a, b) => a.lastActiveAt - b.lastActiveAt)
  const dropped = byLru.slice(0, overflow)
  for (const session of dropped) {
    void deleteSessionBoards(session.id)
    onDrop?.(session)
  }
  const drop = new Set(dropped.map((s) => s.id))
  return sessions.filter((s) => !drop.has(s.id))
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const { user, loading: authLoading } = useAuth()
  const userId = user?.id ?? null
  const userIdRef = useRef(userId)
  userIdRef.current = userId
  const [threadId, setThreadId] = useState("default_session")
  const threadIdRef = useRef(threadId)
  threadIdRef.current = threadId
  const [sessions, setSessions] = useState<SessionRecord[]>([])
  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions
  const [hydrated, setHydrated] = useState(false)
  const [cloudReady, setCloudReady] = useState(false)

  const dropRemote = useCallback((session: SessionRecord) => {
    const uid = userIdRef.current
    if (!uid) return
    void deleteRemoteSession(uid, session.id, session.whiteboards?.map((board) => board.id) ?? [])
  }, [])

  useEffect(() => {
    const stored = loadStored()
    setSessions(stored.sessions)
    const restored = stored.sessions.find((s) => s.id === stored.lastThreadId)
    setThreadId(restored?.id ?? mintThreadId())
    setHydrated(true)
  }, [])

  useEffect(() => {
    if (!hydrated || authLoading) return
    let cancelled = false
    if (!userId) {
      setCloudReady(true)
      return
    }
    setCloudReady(false)
    void (async () => {
      const remote = await fetchRemoteSessions(userId)
      if (cancelled) return
      if (remote) {
        setSessions((prev) => evictIfNeeded(mergeSessions(prev, remote), threadIdRef.current, dropRemote))
      }
      if (!cancelled) setCloudReady(true)
    })()
    return () => {
      cancelled = true
    }
  }, [hydrated, authLoading, userId, dropRemote])

  useEffect(() => {
    if (!hydrated) return
    const committed = sessions.some((s) => s.id === threadId)
    const previous = loadStored()
    saveStored({
      sessions,
      lastThreadId: committed ? threadId : previous.lastThreadId,
    })
  }, [sessions, threadId, hydrated])

  useEffect(() => {
    if (!hydrated || !cloudReady || !userId) return
    const handle = window.setTimeout(() => {
      for (const session of sessions) void upsertSession(userId, session)
    }, 500)
    return () => window.clearTimeout(handle)
  }, [sessions, hydrated, cloudReady, userId])

  const isDraft = useMemo(
    () => !sessions.some((s) => s.id === threadId),
    [sessions, threadId],
  )

  const prepareDraft = useCallback(() => {
    setThreadId(mintThreadId())
  }, [])

  const commitSessionTitle = useCallback(
    (firstPrompt: string, attachmentName?: string) => {
      setSessions((prev) => {
        if (prev.some((s) => s.id === threadId)) {
          return prev.map((s) =>
            s.id === threadId ? { ...s, lastActiveAt: Date.now() } : s,
          )
        }
        const now = Date.now()
        const record: SessionRecord = {
          id: threadId,
          title: titleFromFirstPrompt(firstPrompt, attachmentName),
          createdAt: now,
          lastActiveAt: now,
          messages: [],
          chapters: [],
          activeQuestionId: null,
          deskItems: [],
          activeProblem: null,
          whiteboards: [],
          activeWhiteboardId: null,
        }
        return evictIfNeeded([record, ...prev], threadId, dropRemote)
      })
    },
    [threadId, dropRemote],
  )

  const selectSession = useCallback((id: string) => {
    setSessions((prev) => {
      if (!prev.some((s) => s.id === id)) return prev
      setThreadId(id)
      return prev.map((s) => (s.id === id ? { ...s, lastActiveAt: Date.now() } : s))
    })
  }, [])

  const getSession = useCallback(
    (id: string) => sessions.find((s) => s.id === id),
    [sessions],
  )

  const patchSession = useCallback((id: string, patch: Partial<SessionRecord>) => {
    const now = Date.now()
    const current = sessionsRef.current.find((session) => session.id === id)
    if (!current) return
    const next: SessionRecord = { ...current, ...patch, lastActiveAt: now }
    if (patch.title !== undefined) next.title = patch.title
    sessionsRef.current = sessionsRef.current.map((session) => (session.id === id ? next : session))
    setSessions(sessionsRef.current)
    const uid = userIdRef.current
    if (uid) void upsertSession(uid, next)
  }, [])

  const deleteSession = useCallback(
    (id: string) => {
      const existing = sessions.find((s) => s.id === id)
      void deleteSessionBoards(id)
      if (existing) dropRemote(existing)
      const remaining = sessions.filter((s) => s.id !== id)
      setSessions(remaining)
      if (id === threadId) {
        const fallback = [...remaining].sort((a, b) => b.lastActiveAt - a.lastActiveAt)[0]
        setThreadId(fallback?.id ?? mintThreadId())
      }
    },
    [sessions, threadId, dropRemote],
  )

  const value = useMemo(
    () => ({
      threadId,
      sessions,
      isDraft,
      prepareDraft,
      commitSessionTitle,
      selectSession,
      getSession,
      patchSession,
      deleteSession,
      resetSession: prepareDraft,
      sessionsReady: hydrated && cloudReady,
    }),
    [
      threadId,
      sessions,
      isDraft,
      prepareDraft,
      commitSessionTitle,
      selectSession,
      getSession,
      patchSession,
      deleteSession,
      hydrated,
      cloudReady,
    ],
  )

  if (!hydrated) {
    return <SessionContext.Provider value={value}>{null}</SessionContext.Provider>
  }

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession() {
  return useContext(SessionContext)
}

/** Persist which thread to restore on the next `/` mount (used from Collections). */
export function rememberLastThread(id: string) {
  const stored = loadStored()
  saveStored({ ...stored, lastThreadId: id })
}
