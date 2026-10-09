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
import {
  createTLStore,
  defaultShapeUtils,
  getSnapshot,
  loadSnapshot,
  type TLEditorSnapshot,
  type TLStore,
} from "tldraw"
import {
  deleteBoardSnapshot,
  saveBoardSnapshot,
} from "@/lib/board-idb"
import {
  CLOUD_IDLE_MS,
  deleteBoardObject,
  resolveBoardClock,
  uploadBoardSnapshot,
} from "@/lib/board-cloud"
import { isSyncFrozen, registerFlushHook } from "@/lib/user-data"
import { useAuth } from "../auth-context"
import { useLanguage } from "../language-context"
import { useSession } from "../session-context"
import { useDesk, type DeskItem, type WhiteboardMeta } from "../desk-context"
import { deskShapeUtils } from "./desk-shape-utils"

export const MAX_BOARDS = 6

function makeStore(snapshot?: TLEditorSnapshot | null): TLStore {
  const store = createTLStore({
    shapeUtils: [...defaultShapeUtils, ...deskShapeUtils],
  })
  if (snapshot) {
    try {
      loadSnapshot(store, snapshot)
    } catch {
      /* corrupt / schema mismatch — start empty */
    }
  }
  return store
}

function defaultBoard(name: string): WhiteboardMeta {
  return { id: crypto.randomUUID(), name }
}

type WhiteboardContextValue = {
  boards: WhiteboardMeta[]
  activeBoardId: string
  activeStore: TLStore | null
  ready: boolean
  selectBoard: (id: string) => void
  addBoard: () => void
  closeBoard: (id: string) => void
  renameBoard: (id: string, name: string) => void
}

const WhiteboardContext = createContext<WhiteboardContextValue>({
  boards: [],
  activeBoardId: "",
  activeStore: null,
  ready: false,
  selectBoard: () => {},
  addBoard: () => {},
  closeBoard: () => {},
  renameBoard: () => {},
})

export function WhiteboardProvider({ children }: { children: ReactNode }) {
  const { threadId, isDraft, getSession, sessionsReady, patchSession } = useSession()
  const { session: authSession } = useAuth()
  const userId = authSession?.user?.id ?? null
  const accessToken = authSession?.access_token ?? ""
  const { replaceDeskItems, resetCanvasQueue } = useDesk()
  const { t } = useLanguage()
  const [boards, setBoards] = useState<WhiteboardMeta[]>([])
  const [activeBoardId, setActiveBoardId] = useState("")
  const [ready, setReady] = useState(false)
  const [storeEpoch, setStoreEpoch] = useState(0)
  const storesRef = useRef(new Map<string, TLStore>())
  const boardsRef = useRef<WhiteboardMeta[]>([])
  boardsRef.current = boards
  const storesThreadRef = useRef(threadId)
  const persistTimers = useRef(new Map<string, number>())
  const cloudTimers = useRef(new Map<string, number>())
  const unlistenRef = useRef<Array<() => void>>([])
  const dirtyRef = useRef(new Set<string>())
  const genRef = useRef(new Map<string, number>())
  const inflightRef = useRef(new Set<string>())
  const rerunRef = useRef(new Set<string>())
  const retiredRef = useRef(new Set<string>())
  const userIdRef = useRef(userId)
  userIdRef.current = userId
  const tokenRef = useRef(accessToken)
  tokenRef.current = accessToken
  const threadIdRef = useRef(threadId)
  threadIdRef.current = threadId
  const isDraftRef = useRef(isDraft)
  isDraftRef.current = isDraft

  const markDirty = useCallback((boardId: string) => {
    dirtyRef.current.add(boardId)
    genRef.current.set(boardId, (genRef.current.get(boardId) ?? 0) + 1)
  }, [])

  const stampCloud = useCallback((boardId: string, savedAt: number) => {
    setBoards((prev) =>
      prev.map((board) => (board.id === boardId ? { ...board, cloudSavedAt: savedAt } : board)),
    )
  }, [])

  const clearCloudTimer = useCallback((boardId: string) => {
    const prev = cloudTimers.current.get(boardId)
    if (prev) window.clearTimeout(prev)
    cloudTimers.current.delete(boardId)
  }, [])

  const flushBoardRef = useRef<
    (boardId: string, keepalive: boolean, thread?: string) => void
  >(() => {})

  const scheduleCloud = useCallback(
    (boardId: string) => {
      if (!userIdRef.current || isDraftRef.current) return
      clearCloudTimer(boardId)
      const timer = window.setTimeout(() => {
        cloudTimers.current.delete(boardId)
        flushBoardRef.current(boardId, false)
      }, CLOUD_IDLE_MS)
      cloudTimers.current.set(boardId, timer)
    },
    [clearCloudTimer],
  )

  flushBoardRef.current = (boardId, keepalive, thread = threadIdRef.current) => {
    const uid = userIdRef.current
    const token = tokenRef.current
    const retiredKey = `${thread}:${boardId}`
    if (!uid || !token || isSyncFrozen()) return
    if (retiredRef.current.has(retiredKey)) return
    if (!dirtyRef.current.has(boardId)) return
    if (inflightRef.current.has(boardId) && !keepalive) {
      rerunRef.current.add(boardId)
      return
    }
    const store = storesRef.current.get(boardId)
    if (!store) return
    let snapshot: ReturnType<typeof getSnapshot>
    try {
      snapshot = getSnapshot(store)
    } catch {
      return
    }
    const savedAt = Date.now()
    const gen = genRef.current.get(boardId) ?? 0
    void saveBoardSnapshot(thread, boardId, snapshot, savedAt)
    const pending = uploadBoardSnapshot({
      userId: uid,
      threadId: thread,
      boardId,
      snapshot,
      accessToken: token,
      keepalive,
    })
    const finish = (ok: boolean) => {
      if (retiredRef.current.has(retiredKey)) {
        void deleteBoardObject(uid, thread, boardId)
        return
      }
      if (!ok) {
        rerunRef.current.delete(boardId)
        if (!keepalive) scheduleCloud(boardId)
        return
      }
      if ((genRef.current.get(boardId) ?? 0) !== gen) return
      dirtyRef.current.delete(boardId)
      rerunRef.current.delete(boardId)
      stampCloud(boardId, savedAt)
    }
    if (keepalive) {
      void pending.then(finish)
      return
    }
    inflightRef.current.add(boardId)
    void pending.then(finish).finally(() => {
      inflightRef.current.delete(boardId)
      if (rerunRef.current.has(boardId) && dirtyRef.current.has(boardId)) {
        rerunRef.current.delete(boardId)
        flushBoardRef.current(boardId, false, thread)
      }
    })
  }

  const persistNow = useCallback((boardId: string) => {
    if (isDraftRef.current || isSyncFrozen()) return
    const store = storesRef.current.get(boardId)
    if (!store) return
    try {
      void saveBoardSnapshot(threadIdRef.current, boardId, getSnapshot(store), Date.now())
    } catch {
      /* ignore */
    }
  }, [])

  const schedulePersist = useCallback(
    (boardId: string) => {
      if (isDraftRef.current) return
      markDirty(boardId)
      const prev = persistTimers.current.get(boardId)
      if (prev) window.clearTimeout(prev)
      const timer = window.setTimeout(() => persistNow(boardId), 500)
      persistTimers.current.set(boardId, timer)
      scheduleCloud(boardId)
    },
    [markDirty, persistNow, scheduleCloud],
  )

  const attachListeners = useCallback(
    (map: Map<string, TLStore>) => {
      for (const stop of unlistenRef.current) stop()
      unlistenRef.current = []
      for (const [boardId, store] of map) {
        const stop = store.listen(
          () => schedulePersist(boardId),
          { scope: "all", source: "user" },
        )
        unlistenRef.current.push(stop)
      }
    },
    [schedulePersist],
  )

  const getSessionRef = useRef(getSession)
  getSessionRef.current = getSession
  const tRef = useRef(t)
  tRef.current = t
  const attachListenersRef = useRef(attachListeners)
  attachListenersRef.current = attachListeners
  const scheduleCloudRef = useRef(scheduleCloud)
  scheduleCloudRef.current = scheduleCloud
  const replaceDeskItemsRef = useRef(replaceDeskItems)
  replaceDeskItemsRef.current = replaceDeskItems
  const resetCanvasQueueRef = useRef(resetCanvasQueue)
  resetCanvasQueueRef.current = resetCanvasQueue
  const wasDraftRef = useRef(isDraft)

  const clearBoardTimers = useCallback((boardId?: string) => {
    const persistIds = boardId ? [boardId] : [...persistTimers.current.keys()]
    for (const id of persistIds) {
      const timer = persistTimers.current.get(id)
      if (timer) window.clearTimeout(timer)
      persistTimers.current.delete(id)
    }
    const cloudIds = boardId ? [boardId] : [...cloudTimers.current.keys()]
    for (const id of cloudIds) {
      const timer = cloudTimers.current.get(id)
      if (timer) window.clearTimeout(timer)
      cloudTimers.current.delete(id)
    }
  }, [])

  useEffect(() => {
    if (!sessionsReady) return
    let cancelled = false
    const activeThread = threadId

    if (storesThreadRef.current === activeThread && storesRef.current.size > 0) {
      attachListenersRef.current(storesRef.current)
      setReady(true)
      return () => {
        cancelled = true
      }
    }

    async function hydrate() {
      const keepMounted = storesRef.current.size > 0
      if (!keepMounted) setReady(false)
      clearBoardTimers()
      for (const stop of unlistenRef.current) stop()
      unlistenRef.current = []

      const rec = getSessionRef.current(activeThread)
      const fallbackName = tRef.current("desk.boardName", { n: 1 })
      const sameThreadBoards =
        storesThreadRef.current === activeThread ? boardsRef.current : []
      const metas: WhiteboardMeta[] =
        rec?.whiteboards && rec.whiteboards.length > 0
          ? rec.whiteboards
          : sameThreadBoards.length > 0
            ? sameThreadBoards
            : [defaultBoard(fallbackName)]
      const active =
        rec?.activeWhiteboardId && metas.some((board) => board.id === rec.activeWhiteboardId)
          ? rec.activeWhiteboardId
          : metas[0].id

      const sameThread = storesThreadRef.current === activeThread
      const next = new Map<string, TLStore>()
      const nextDirty = new Set<string>()
      let migratingItems: DeskItem[] | null = null
      for (const board of metas) {
        let pending: ReturnType<typeof getSnapshot> | null = null
        if (sameThread && dirtyRef.current.has(board.id)) {
          const live = storesRef.current.get(board.id)
          if (live) {
            try {
              pending = getSnapshot(live)
            } catch {
              pending = null
            }
          }
        }
        const resolved = await resolveBoardClock({
          userId: userIdRef.current,
          threadId: activeThread,
          boardId: board.id,
          cloudSavedAt: board.cloudSavedAt ?? 0,
          pendingSnapshot: pending,
        })
        if (cancelled) return
        const snap = (resolved.snapshot ?? null) as TLEditorSnapshot | null
        const live = storesRef.current.get(board.id)
        next.set(board.id, live ?? makeStore(snap))
        if (resolved.dirty) nextDirty.add(board.id)
        if (!snap && board.id === active && rec?.deskItems && rec.deskItems.length > 0) {
          migratingItems = rec.deskItems
        }
      }

      if (cancelled) return
      const previousActive = storesRef.current.get(active)
      const nextActive = next.get(active)
      dirtyRef.current = nextDirty
      storesThreadRef.current = activeThread
      storesRef.current = next
      attachListenersRef.current(next)
      setBoards(metas)
      setActiveBoardId(active)
      if (!previousActive || previousActive !== nextActive) setStoreEpoch((n) => n + 1)
      if (migratingItems) replaceDeskItemsRef.current(migratingItems)
      else resetCanvasQueueRef.current()
      setReady(true)
      for (const boardId of nextDirty) scheduleCloudRef.current(boardId)
    }

    void hydrate()
    return () => {
      cancelled = true
      clearBoardTimers()
      for (const stop of unlistenRef.current) stop()
      unlistenRef.current = []
    }
  }, [threadId, sessionsReady, clearBoardTimers])

  useEffect(() => {
    const thread = threadId
    return () => {
      for (const boardId of storesRef.current.keys()) {
        if (!dirtyRef.current.has(boardId)) continue
        flushBoardRef.current(boardId, false, thread)
      }
    }
  }, [threadId])

  useEffect(
    () =>
      registerFlushHook(() => {
        for (const boardId of storesRef.current.keys()) {
          flushBoardRef.current(boardId, false)
        }
      }),
    [],
  )

  useEffect(() => {
    const onExit = () => {
      for (const boardId of storesRef.current.keys()) {
        flushBoardRef.current(boardId, true)
      }
    }
    const onVisibility = () => {
      if (document.visibilityState === "hidden") onExit()
    }
    document.addEventListener("visibilitychange", onVisibility)
    window.addEventListener("pagehide", onExit)
    return () => {
      document.removeEventListener("visibilitychange", onVisibility)
      window.removeEventListener("pagehide", onExit)
    }
  }, [])

  useEffect(() => {
    if (!ready) return
    const becameCommitted = wasDraftRef.current && !isDraft
    wasDraftRef.current = isDraft
    if (isDraft) return
    attachListeners(storesRef.current)
    if (!becameCommitted) return
    for (const boardId of storesRef.current.keys()) {
      markDirty(boardId)
      persistNow(boardId)
      scheduleCloud(boardId)
    }
  }, [isDraft, ready, persistNow, attachListeners, markDirty, scheduleCloud])

  const renameBoard = useCallback((id: string, name: string) => {
    const nextName = name.trim()
    if (!nextName) return
    const nextBoards = boards.map((board) =>
      board.id === id && board.name !== nextName ? { ...board, name: nextName } : board,
    )
    if (nextBoards.every((board, index) => board === boards[index])) return
    setBoards(nextBoards)
    if (!isDraftRef.current) {
      patchSession(threadIdRef.current, { whiteboards: nextBoards })
    }
  }, [boards, patchSession])

  const selectBoard = useCallback(
    (id: string) => {
      if (id === activeBoardId) return
      persistNow(activeBoardId)
      flushBoardRef.current(activeBoardId, false)
      setActiveBoardId(id)
    },
    [activeBoardId, persistNow],
  )

  const addBoard = useCallback(() => {
    if (boards.length >= MAX_BOARDS) return
    const board = defaultBoard(t("desk.boardName", { n: boards.length + 1 }))
    const store = makeStore()
    storesRef.current.set(board.id, store)
    attachListeners(storesRef.current)
    setBoards((prev) => [...prev, board])
    setActiveBoardId(board.id)
    setStoreEpoch((n) => n + 1)
  }, [boards, t, attachListeners])

  const closeBoard = useCallback(
    (id: string) => {
      setBoards((prev) => {
        if (prev.length <= 1) return prev
        const next = prev.filter((board) => board.id !== id)
        retiredRef.current.add(`${threadId}:${id}`)
        dirtyRef.current.delete(id)
        clearBoardTimers(id)
        storesRef.current.delete(id)
        void deleteBoardSnapshot(threadId, id)
        const uid = userIdRef.current
        if (uid) void deleteBoardObject(uid, threadId, id)
        const fallback = next[0]?.id
        if (id === activeBoardId && fallback) setActiveBoardId(fallback)
        setStoreEpoch((n) => n + 1)
        return next
      })
    },
    [activeBoardId, clearBoardTimers, threadId],
  )

  const activeStore = useMemo(() => {
    void storeEpoch
    return storesRef.current.get(activeBoardId) ?? null
  }, [activeBoardId, storeEpoch])

  const value = useMemo(
    () => ({
      boards,
      activeBoardId,
      activeStore,
      ready,
      selectBoard,
      addBoard,
      closeBoard,
      renameBoard,
    }),
    [boards, activeBoardId, activeStore, ready, selectBoard, addBoard, closeBoard, renameBoard],
  )

  return <WhiteboardContext.Provider value={value}>{children}</WhiteboardContext.Provider>
}

export function useWhiteboard() {
  return useContext(WhiteboardContext)
}
