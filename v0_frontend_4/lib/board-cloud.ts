import { loadBoardRecord, saveBoardSnapshot, type BoardSnapshot } from "@/lib/board-idb"
import { getSupabasePublicEnv } from "@/lib/supabase/env"
import { createClient } from "@/lib/supabase/client"
import { isSyncFrozen, trackPending } from "@/lib/user-data"

export const WHITEBOARD_BUCKET = "whiteboards"
/** Chrome drops keepalive bodies above this size. */
export const KEEPALIVE_LIMIT = 64 * 1024
export const CLOUD_IDLE_MS = 45_000

export function boardObjectPath(userId: string, threadId: string, boardId: string) {
  return `${userId}/${threadId}/${boardId}`
}

function encodedPath(path: string) {
  return path.split("/").map(encodeURIComponent).join("/")
}

export async function uploadBoardSnapshot(options: {
  userId: string
  threadId: string
  boardId: string
  snapshot: BoardSnapshot
  accessToken: string
  keepalive: boolean
}): Promise<boolean> {
  const env = getSupabasePublicEnv()
  if (!env || !options.accessToken || isSyncFrozen()) return false
  const body = JSON.stringify(options.snapshot)
  if (options.keepalive && new TextEncoder().encode(body).length > KEEPALIVE_LIMIT) {
    console.warn("[Boards] Exit upload skipped; snapshot exceeds keepalive limit")
    return false
  }
  const path = encodedPath(boardObjectPath(options.userId, options.threadId, options.boardId))
  return trackPending(
    fetch(`${env.url}/storage/v1/object/${WHITEBOARD_BUCKET}/${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${options.accessToken}`,
        apikey: env.anonKey,
        "Content-Type": "application/json",
        "x-upsert": "true",
      },
      body,
      keepalive: options.keepalive,
    })
      .then((response) => {
        if (!response.ok) {
          console.error("[Boards] Upload failed", response.status)
          return false
        }
        return true
      })
      .catch((error: unknown) => {
        console.error("[Boards] Upload failed", error)
        return false
      }),
  )
}

export async function downloadBoardSnapshot(
  userId: string,
  threadId: string,
  boardId: string,
): Promise<{ status: "ok"; snapshot: BoardSnapshot } | { status: "missing" } | { status: "error" }> {
  const supabase = createClient()
  if (!supabase) return { status: "error" }
  const { data, error } = await supabase.storage
    .from(WHITEBOARD_BUCKET)
    .download(boardObjectPath(userId, threadId, boardId))
  if (error || !data) {
    if (error && /not found|404/i.test(error.message)) return { status: "missing" }
    if (error) console.error("[Boards] Download failed", error.message)
    return { status: "error" }
  }
  try {
    return { status: "ok", snapshot: JSON.parse(await data.text()) as BoardSnapshot }
  } catch {
    return { status: "error" }
  }
}

export async function deleteBoardObject(userId: string, threadId: string, boardId: string) {
  const supabase = createClient()
  if (!supabase) return
  const { error } = await supabase.storage
    .from(WHITEBOARD_BUCKET)
    .remove([boardObjectPath(userId, threadId, boardId)])
  if (error) console.error("[Boards] Delete failed", error.message)
}

/**
 * Compare cloud and local clocks before the editor listens.
 * A newer cloud snapshot replaces IndexedDB. A newer or only-local snapshot stays and is dirty.
 */
export async function resolveBoardClock(options: {
  userId: string | null
  threadId: string
  boardId: string
  cloudSavedAt: number
  pendingSnapshot?: BoardSnapshot | null
}): Promise<{ snapshot: BoardSnapshot | null; dirty: boolean }> {
  const local = await loadBoardRecord(options.threadId, options.boardId)
  const localAt = local?.localSavedAt ?? 0
  const cloudAt = options.cloudSavedAt
  const hasLocal = Boolean(local)
  const hasCloud = cloudAt > 0

  if (options.pendingSnapshot && options.userId && localAt >= cloudAt) {
    const savedAt = Date.now()
    await saveBoardSnapshot(options.threadId, options.boardId, options.pendingSnapshot, savedAt)
    return { snapshot: options.pendingSnapshot, dirty: true }
  }

  if (!options.userId || !hasCloud) {
    return { snapshot: local?.snapshot ?? null, dirty: Boolean(options.userId && hasLocal && !hasCloud) }
  }

  if (cloudAt > localAt || !hasLocal) {
    const remote = await downloadBoardSnapshot(options.userId, options.threadId, options.boardId)
    if (remote.status === "ok") {
      await saveBoardSnapshot(options.threadId, options.boardId, remote.snapshot, cloudAt)
      return { snapshot: remote.snapshot, dirty: false }
    }
    if (remote.status === "missing" && hasLocal) {
      return { snapshot: local?.snapshot ?? null, dirty: true }
    }
    return { snapshot: local?.snapshot ?? null, dirty: false }
  }

  if (localAt > cloudAt) {
    return { snapshot: local?.snapshot ?? null, dirty: true }
  }

  return { snapshot: local?.snapshot ?? null, dirty: false }
}
