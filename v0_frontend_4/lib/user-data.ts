import { deleteAllBoards } from "@/lib/board-idb"

/** Per-user browser data. Theme, chat width and language are device settings and stay. */
export const SESSIONS_KEY = "locknlearn:sessions:v1"
export const FORMULAS_KEY = "locknlearn:savedFormulas"
export const GRAPHS_KEY = "locknlearn:savedGraphs"
export const STRUGGLES_KEY = "locknlearn:savedStruggles"
export const BOOKMARKS_KEY = "locknlearn:savedBookmarks"
export const OWNER_KEY = "locknlearn:owner"
export const DOCUMENT_PATH_PREFIX = "locknlearn.documentPath."

const USER_LOCAL_KEYS = [SESSIONS_KEY, FORMULAS_KEY, GRAPHS_KEY, STRUGGLES_KEY, BOOKMARKS_KEY, OWNER_KEY]

export function readOwner(): string | null {
  try {
    return window.localStorage.getItem(OWNER_KEY) || null
  } catch {
    return null
  }
}

export function writeOwner(userId: string) {
  try {
    window.localStorage.setItem(OWNER_KEY, userId)
  } catch {
    /* private browsing */
  }
}

function removeSessionStoragePrefix(prefix: string) {
  try {
    const doomed: string[] = []
    for (let i = 0; i < window.sessionStorage.length; i += 1) {
      const key = window.sessionStorage.key(i)
      if (key && key.startsWith(prefix)) doomed.push(key)
    }
    for (const key of doomed) window.sessionStorage.removeItem(key)
  } catch {
    /* private browsing */
  }
}

/** Removes every key that belongs to a signed-in user, including the owner marker. */
export async function wipeLocalUserData() {
  for (const key of USER_LOCAL_KEYS) {
    try {
      window.localStorage.removeItem(key)
    } catch {
      /* private browsing */
    }
  }
  removeSessionStoragePrefix(DOCUMENT_PATH_PREFIX)
  await deleteAllBoards()
}

const pending = new Set<Promise<unknown>>()
const flushHooks = new Set<() => void>()
let frozen = false

/** Registers an in-flight cloud write so sign-out can wait for it. */
export function trackPending<T>(work: Promise<T>): Promise<T> {
  pending.add(work)
  const settle = () => {
    pending.delete(work)
  }
  work.then(settle, settle)
  return work
}

export function setSyncFrozen(value: boolean) {
  frozen = value
}

export function isSyncFrozen() {
  return frozen
}

/** Providers register a callback that starts uploads for their unsaved data. */
export function registerFlushHook(hook: () => void) {
  flushHooks.add(hook)
  return () => {
    flushHooks.delete(hook)
  }
}

/**
 * Starts every registered flush, then waits for in-flight writes. Resolves after
 * `timeoutMs` even when the network is slow, so callers never hang.
 */
export async function flushPendingWithin(timeoutMs: number) {
  for (const hook of flushHooks) {
    try {
      hook()
    } catch {
      /* a failing hook must not block sign-out */
    }
  }
  let expired = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<void>((resolve) => {
    timer = setTimeout(() => {
      expired = true
      resolve()
    }, timeoutMs)
  })
  const drain = (async () => {
    while (!expired && pending.size > 0) {
      await Promise.allSettled([...pending])
    }
  })()
  await Promise.race([drain, deadline])
  if (timer) clearTimeout(timer)
}
