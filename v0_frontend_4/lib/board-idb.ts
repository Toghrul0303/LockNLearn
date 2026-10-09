/** IndexedDB snapshots for tldraw whiteboards. Session metadata stays in localStorage. */

const DB_NAME = "locknlearn-boards"
const DB_VERSION = 1
const STORE = "snapshots"

export type BoardSnapshot = unknown

export type BoardDiskRecord = {
  snapshot: BoardSnapshot
  localSavedAt: number
}

function asDiskRecord(raw: unknown): BoardDiskRecord | null {
  if (!raw || typeof raw !== "object") return null
  const record = raw as { snapshot?: BoardSnapshot; localSavedAt?: unknown }
  if ("snapshot" in record && typeof record.localSavedAt === "number") {
    return { snapshot: record.snapshot, localSavedAt: record.localSavedAt }
  }
  return { snapshot: raw, localSavedAt: 0 }
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export function boardStorageKey(threadId: string, boardId: string) {
  return `${threadId}:${boardId}`
}

export async function loadBoardRecord(
  threadId: string,
  boardId: string,
): Promise<BoardDiskRecord | null> {
  if (typeof indexedDB === "undefined") return null
  const db = await openDb()
  try {
    const raw = await new Promise<unknown>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly")
      const req = tx.objectStore(STORE).get(boardStorageKey(threadId, boardId))
      req.onsuccess = () => resolve(req.result ?? null)
      req.onerror = () => reject(req.error)
    })
    return asDiskRecord(raw)
  } catch {
    return null
  } finally {
    db.close()
  }
}

export async function loadBoardSnapshot(
  threadId: string,
  boardId: string,
): Promise<BoardSnapshot | null> {
  const record = await loadBoardRecord(threadId, boardId)
  return record?.snapshot ?? null
}

export async function saveBoardSnapshot(
  threadId: string,
  boardId: string,
  snapshot: BoardSnapshot,
  localSavedAt = Date.now(),
): Promise<void> {
  if (typeof indexedDB === "undefined") return
  const db = await openDb()
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite")
      const record: BoardDiskRecord = { snapshot, localSavedAt }
      tx.objectStore(STORE).put(record, boardStorageKey(threadId, boardId))
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* quota / private browsing */
  } finally {
    db.close()
  }
}

export async function deleteBoardSnapshot(threadId: string, boardId: string): Promise<void> {
  if (typeof indexedDB === "undefined") return
  const db = await openDb()
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite")
      tx.objectStore(STORE).delete(boardStorageKey(threadId, boardId))
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* ignore */
  } finally {
    db.close()
  }
}

export async function deleteAllBoards(): Promise<void> {
  if (typeof indexedDB === "undefined") return
  await new Promise<void>((resolve) => {
    const settle = () => resolve()
    // A blocked delete completes once the other connections close; do not wait on it.
    const timer = window.setTimeout(settle, 1000)
    const req = indexedDB.deleteDatabase(DB_NAME)
    req.onsuccess = () => {
      window.clearTimeout(timer)
      settle()
    }
    req.onerror = () => {
      window.clearTimeout(timer)
      settle()
    }
    req.onblocked = () => {
      /* the 1s timer resolves; the browser finishes the delete on its own */
    }
  })
}

export async function deleteSessionBoards(threadId: string): Promise<void> {
  if (typeof indexedDB === "undefined") return
  const db = await openDb()
  const prefix = `${threadId}:`
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite")
      const store = tx.objectStore(STORE)
      const req = store.openCursor()
      req.onsuccess = () => {
        const cursor = req.result
        if (!cursor) return
        if (typeof cursor.key === "string" && cursor.key.startsWith(prefix)) {
          cursor.delete()
        }
        cursor.continue()
      }
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* ignore */
  } finally {
    db.close()
  }
}
