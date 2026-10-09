import type { SessionRecord } from "@/components/workspace/session-context"
import { createClient } from "@/lib/supabase/client"
import { WHITEBOARD_BUCKET } from "@/lib/board-cloud"
import { isSyncFrozen, trackPending } from "@/lib/user-data"

function iso(savedAt: number) {
  return new Date(savedAt).toISOString()
}

function ms(value: string | null | undefined) {
  const n = value ? Date.parse(value) : NaN
  return Number.isFinite(n) ? n : 0
}

function logRemoteError(label: string, error: { message: string } | null) {
  if (error) console.error(`[Sessions] ${label}`, error.message)
}

/** Last ticket that superseded an in-flight upsert. A later delete wins. */
const supersededAt = new Map<string, number>()
let ticketClock = 0

function rowKey(userId: string, id: string) {
  return `${userId}:${id}`
}

function asSession(row: {
  id: string
  title: string
  payload: unknown
  updated_at: string
}): SessionRecord | null {
  if (!row.payload || typeof row.payload !== "object") return null
  const payload = row.payload as Partial<SessionRecord>
  const updatedAt = ms(row.updated_at)
  return {
    id: row.id,
    title: row.title || payload.title || "New session",
    createdAt: payload.createdAt || updatedAt || Date.now(),
    lastActiveAt: updatedAt || payload.lastActiveAt || 0,
    messages: Array.isArray(payload.messages) ? payload.messages : [],
    chapters: Array.isArray(payload.chapters) ? payload.chapters : [],
    activeQuestionId: payload.activeQuestionId ?? null,
    deskItems: Array.isArray(payload.deskItems) ? payload.deskItems : [],
    activeProblem: payload.activeProblem ?? null,
    whiteboards: Array.isArray(payload.whiteboards) ? payload.whiteboards : [],
    activeWhiteboardId: payload.activeWhiteboardId ?? null,
    documentPath: payload.documentPath ?? null,
  }
}

/** Last-write-wins on `lastActiveAt` / `updated_at`. Equal clocks keep the local row. */
export function mergeSessions(local: SessionRecord[], remote: SessionRecord[]): SessionRecord[] {
  const map = new Map<string, SessionRecord>()
  for (const item of local) map.set(item.id, item)
  for (const item of remote) {
    const existing = map.get(item.id)
    if (!existing || item.lastActiveAt > existing.lastActiveAt) map.set(item.id, item)
  }
  return [...map.values()]
}

export async function fetchRemoteSessions(userId: string): Promise<SessionRecord[] | null> {
  const supabase = createClient()
  if (!supabase) return null
  const { data, error } = await supabase
    .from("sessions")
    .select("id, title, payload, updated_at")
    .eq("user_id", userId)
  if (error) {
    logRemoteError("fetch", error)
    return null
  }
  return (data ?? []).flatMap((row) => {
    const session = asSession(row as {
      id: string
      title: string
      payload: unknown
      updated_at: string
    })
    return session ? [session] : []
  })
}

export function upsertSession(userId: string, session: SessionRecord): Promise<void> {
  if (isSyncFrozen()) return Promise.resolve()
  return trackPending(runUpsertSession(userId, session))
}

async function runUpsertSession(userId: string, session: SessionRecord) {
  const supabase = createClient()
  if (!supabase) return
  const key = rowKey(userId, session.id)
  const ticket = supersededAt.get(key) ?? 0
  const payload: SessionRecord = { ...session, title: session.title }
  const { error } = await supabase.from("sessions").upsert(
    {
      id: session.id,
      user_id: userId,
      title: payload.title,
      payload,
      updated_at: iso(payload.lastActiveAt),
    },
    { onConflict: "user_id,id" },
  )
  logRemoteError("upsert", error)
  if ((supersededAt.get(key) ?? 0) !== ticket) {
    const { error: deleteError } = await supabase
      .from("sessions")
      .delete()
      .eq("user_id", userId)
      .eq("id", session.id)
    logRemoteError("upsert superseded", deleteError)
  }
}

export async function deleteRemoteSession(userId: string, sessionId: string, boardIds: string[]) {
  const supabase = createClient()
  if (!supabase) return
  const key = rowKey(userId, sessionId)
  supersededAt.set(key, ++ticketClock)
  if (boardIds.length > 0) {
    const paths = boardIds.map((boardId) => `${userId}/${sessionId}/${boardId}`)
    const { error: storageError } = await supabase.storage.from(WHITEBOARD_BUCKET).remove(paths)
    logRemoteError("board delete", storageError)
  }
  const { error } = await supabase.from("sessions").delete().eq("user_id", userId).eq("id", sessionId)
  logRemoteError("delete", error)
}
