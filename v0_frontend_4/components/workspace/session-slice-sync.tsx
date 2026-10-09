"use client"

import { useEffect, useRef } from "react"
import { useDesk } from "./desk-context"
import { useSession } from "./session-context"
import { useTaskTracker } from "./task-tracker-context"
import { useWhiteboard } from "./canvas/whiteboard-context"

/**
 * Keeps Task Tracker + Desk snapshots in the session record, and hydrates
 * both when the student switches threads (or opens a fresh draft).
 */
export function SessionSliceSync() {
  const { threadId, isDraft, getSession, patchSession, sessionsReady } = useSession()
  const { chapters, activeQuestionId, hydrateTracker } = useTaskTracker()
  const { deskItems, activeProblem, setActiveProblem } = useDesk()
  const { boards, activeBoardId, ready } = useWhiteboard()
  const skipPersist = useRef(true)
  const lastHydrated = useRef<string | null>(null)

  useEffect(() => {
    if (!sessionsReady) {
      lastHydrated.current = null
      skipPersist.current = true
      return
    }
    if (lastHydrated.current === threadId) return
    lastHydrated.current = threadId
    skipPersist.current = true
    const rec = getSession(threadId)
    hydrateTracker(rec?.chapters ?? [], rec?.activeQuestionId ?? null)
    setActiveProblem(rec?.activeProblem ?? null)
  }, [threadId, sessionsReady, getSession, hydrateTracker, setActiveProblem])

  useEffect(() => {
    if (!sessionsReady) return
    if (skipPersist.current) {
      skipPersist.current = false
      return
    }
    if (isDraft || !ready) return
    patchSession(threadId, {
      chapters,
      activeQuestionId,
      deskItems,
      activeProblem,
      whiteboards: boards,
      activeWhiteboardId: activeBoardId || null,
    })
  }, [
    chapters,
    activeQuestionId,
    deskItems,
    activeProblem,
    boards,
    activeBoardId,
    isDraft,
    threadId,
    patchSession,
    sessionsReady,
    ready,
  ])

  return null
}
